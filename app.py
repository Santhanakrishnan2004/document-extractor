"""
AI Document Data Extractor
---------------------------
Upload a PDF (invoice, receipt, resume, etc.) and this app uses an LLM
(via Groq's free API) to pull out structured fields and let you download
the result as CSV / Excel.

This is the demo/portfolio piece for freelance gigs (Fiverr etc.) selling
"AI document extraction" / "AI agent" services.

Run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY="your_key_here"
    streamlit run app.py
"""

import io
import json
import os
import traceback

import pandas as pd
import pdfplumber
import streamlit as st
from groq import Groq

MAX_EXTRACTIONS_PER_SESSION = (
    5  # protects your free-tier Groq quota from one visitor/bot
)

SAMPLE_INVOICE_TEXT = """INVOICE

Brightline Office Supplies Co.
482 Kestrel Avenue, Austin, TX 78701
billing@brightlineoffice.com

Invoice Number: INV-20458
Invoice Date: 2026-08-14
Due Date: 2026-09-13

Bill To:
Harper & Voss Consulting
221 Baker Street, Suite 4B
Austin, TX 78702

Description                          Qty     Unit Price     Amount
--------------------------------------------------------------------
A4 Printer Paper (Box of 5 reams)     10        $9.50        $95.00
Black Toner Cartridge (HP 26A)          3       $89.00       $267.00
Wireless Mouse - Ergonomic              5       $18.75        $93.75
Standing Desk Mat                       2       $42.00        $84.00

                                                Subtotal:     $539.75
                                                Sales Tax (8.25%): $44.53
                                                Total Due:    $584.28

Currency: USD
Payment Terms: Net 30
Thank you for your business!
"""

# ---------------------------------------------------------------------
# CONFIG — change this schema to whatever document type you're demoing
# ---------------------------------------------------------------------
DOCUMENT_TYPE = "invoice"  # e.g. "invoice", "receipt", "resume"

EXTRACTION_SCHEMA = {
    "vendor_name": "string - name of the company/person who issued the document",
    "document_number": "string - invoice/receipt number if present",
    "date": "string - document date, format YYYY-MM-DD if possible",
    "total_amount": "number - the final total amount",
    "currency": "string - currency code or symbol, e.g. INR, USD, $",
    "line_items": "array of objects, each with 'description', 'quantity', 'unit_price', 'amount'",
}

MODEL = "openai/gpt-oss-120b"  # fast + free-tier friendly on Groq

# ---------------------------------------------------------------------
# BRANDING — edit these before sending the link to a prospect
# ---------------------------------------------------------------------
FREELANCER_NAME = "Santhana Krishnan"
CONTACT_LINK = "santhanakrishnan9704@gmail.com"

# ---------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------


def get_shared_api_key() -> str:
    """Looks for a demo API key you've configured (Streamlit secrets or env
    var) so prospects can try the demo with zero setup. Returns "" if none
    is configured, in which case visitors must supply their own key."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def extract_text_from_pdf(uploaded_file) -> str:
    """Pulls raw text out of an uploaded PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def build_prompt(document_text: str) -> str:
    schema_str = json.dumps(EXTRACTION_SCHEMA, indent=2)
    return f"""You are a data extraction engine. Extract structured data from the
{DOCUMENT_TYPE} text below and return ONLY a valid JSON object matching this schema
(use null for any field you cannot find, do not invent data):

SCHEMA:
{schema_str}

DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"

Return ONLY the JSON object, no markdown fences, no explanation."""


def call_groq(client: Groq, document_text: str) -> dict:
    prompt = build_prompt(document_text)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def flatten_for_table(data: dict) -> pd.DataFrame:
    """Turns the extracted JSON into a flat table for display/download.
    Line items become their own rows; header fields are repeated."""
    header = {k: v for k, v in data.items() if k != "line_items"}
    line_items = data.get("line_items") or []

    if not line_items:
        return pd.DataFrame([header])

    rows = []
    for item in line_items:
        row = {**header, **item}
        rows.append(row)
    return pd.DataFrame(rows)


def to_xero_bills_csv(
    data: dict,
    account_code: str,
    tax_type: str,
    due_days: int,
    date_fmt: str,
) -> pd.DataFrame:
    """Maps extracted invoice data onto Xero's 'Import Bills' CSV template
    (Business > Bills to pay > Import). Columns with a leading * are the
    ones Xero requires; header names/order must match exactly."""
    vendor = data.get("vendor_name") or "Unknown Supplier"
    invoice_number = data.get("document_number") or ""
    currency = data.get("currency") or ""

    invoice_date = pd.to_datetime(data.get("date"), errors="coerce")
    if pd.isna(invoice_date):
        invoice_date = pd.Timestamp.today()
    due_date = invoice_date + pd.Timedelta(days=due_days)

    line_items = data.get("line_items") or []
    if not line_items:
        line_items = [
            {
                "description": vendor,
                "quantity": 1,
                "unit_price": data.get("total_amount") or 0,
            }
        ]

    rows = []
    for item in line_items:
        unit_amount = item.get("unit_price")
        if unit_amount is None:
            unit_amount = item.get("amount") or 0
        rows.append(
            {
                "*ContactName": vendor,
                "*InvoiceNumber": invoice_number,
                "*InvoiceDate": invoice_date.strftime(date_fmt),
                "*DueDate": due_date.strftime(date_fmt),
                "Description": item.get("description") or vendor,
                "*Quantity": item.get("quantity") or 1,
                "*UnitAmount": unit_amount,
                "*AccountCode": account_code,
                "*TaxType": tax_type,
                "Currency": currency,
            }
        )
    return pd.DataFrame(rows)


def to_quickbooks_csv(
    data: dict,
    category: str,
    terms: str,
    due_days: int,
    date_fmt: str,
) -> pd.DataFrame:
    """Maps extracted invoice data onto the column layout used by the
    common QuickBooks Online bulk-import apps (SaasAnt Transactions,
    Transaction Pro Importer) for Bills. QBO itself has no native generic
    CSV-import screen for bills, so one of these apps is the standard route
    for getting a CSV like this into a client's QuickBooks."""
    vendor = data.get("vendor_name") or "Unknown Vendor"
    bill_no = data.get("document_number") or ""
    currency = data.get("currency") or ""

    bill_date = pd.to_datetime(data.get("date"), errors="coerce")
    if pd.isna(bill_date):
        bill_date = pd.Timestamp.today()
    due_date = bill_date + pd.Timedelta(days=due_days)

    line_items = data.get("line_items") or []
    if not line_items:
        line_items = [
            {
                "description": vendor,
                "quantity": 1,
                "unit_price": data.get("total_amount") or 0,
            }
        ]

    rows = []
    for item in line_items:
        rate = item.get("unit_price")
        if rate is None:
            rate = item.get("amount") or 0
        rows.append(
            {
                "*VendorName": vendor,
                "*BillNo": bill_no,
                "*BillDate": bill_date.strftime(date_fmt),
                "*DueDate": due_date.strftime(date_fmt),
                "Terms": terms,
                "Category": category,
                "*Description": item.get("description") or vendor,
                "*Qty": item.get("quantity") or 1,
                "*Rate": rate,
                "Currency": currency,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Document Data Extractor",
    page_icon="📄",
    layout="wide",
)

if "extraction_count" not in st.session_state:
    st.session_state.extraction_count = 0
if "result" not in st.session_state:
    st.session_state.result = None

shared_api_key = get_shared_api_key()

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.title("📄 AI Document Data Extractor")
st.caption(
    "Upload an invoice PDF and AI pulls out the structured data — then export "
    "it ready-to-import into **Xero** or **QuickBooks**, no manual re-typing."
)

step1, step2, step3 = st.columns(3)
step1.info("**1. Upload**\n\nDrop in an invoice or receipt PDF (or try the sample).")
step2.info(
    "**2. AI extracts**\n\nVendor, line items, dates, and totals — pulled automatically."
)
step3.info(
    "**3. Export**\n\nDownload a CSV/Excel, or one formatted for Xero or QuickBooks."
)

st.markdown("---")

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    if shared_api_key:
        st.success("Demo key active — no setup needed, just try it below.")
        with st.expander("Use your own Groq API key instead"):
            own_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Get a free key at console.groq.com",
            )
        api_key_input = own_key or shared_api_key
    else:
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at console.groq.com",
        )
        st.markdown("[Get a free Groq API key →](https://console.groq.com/keys)")

    st.header("📤 Export settings")
    st.caption("Match these to the client's accounting setup before importing.")

    with st.expander("🟦 Xero", expanded=True):
        xero_account_code = st.text_input("Account code", value="400")
        xero_tax_type = st.text_input(
            "Tax type (exact name from client's Xero)", value="Tax Exempt"
        )
        xero_due_days = st.number_input(
            "Due date = invoice date +",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            key="xero_due_days",
        )
        xero_date_fmt = st.selectbox(
            "Date format", ["DD/MM/YYYY", "MM/DD/YYYY"], index=0, key="xero_date_fmt"
        )

    with st.expander("🟩 QuickBooks"):
        qb_category = st.text_input("Expense category/account", value="Office Supplies")
        qb_terms = st.text_input("Payment terms", value="Net 30")
        qb_due_days = st.number_input(
            "Due date = invoice date +",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            key="qb_due_days",
        )
        qb_date_fmt = st.selectbox(
            "Date format", ["MM/DD/YYYY", "DD/MM/YYYY"], index=0, key="qb_date_fmt"
        )

    if st.session_state.result:
        st.markdown("---")
        if st.button("🔄 Start over / try another document", use_container_width=True):
            st.session_state.result = None
            st.rerun()

    st.markdown("---")
    st.caption(f"Built by **{FREELANCER_NAME}** · [Get in touch]({CONTACT_LINK})")

col_upload, col_sample = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("🧾 Try a sample invoice", use_container_width=True)

is_sample = False
if use_sample:
    doc_text = SAMPLE_INVOICE_TEXT
    doc_ready_to_run = True
    is_sample = True
elif uploaded_file:
    with st.spinner("Reading PDF..."):
        doc_text = extract_text_from_pdf(uploaded_file)
    if not doc_text.strip():
        st.error(
            "Couldn't extract any text from this PDF. It might be a scanned image — OCR support can be added."
        )
        doc_text = None
    doc_ready_to_run = False
else:
    doc_text = None
    doc_ready_to_run = False

if doc_text and not api_key_input:
    st.warning("Add your Groq API key in the sidebar to run extraction.")
elif doc_text:
    if not is_sample:
        with st.expander("Show extracted raw text"):
            st.text(doc_text[:3000])

    run_clicked = doc_ready_to_run or st.button(
        "Extract structured data", type="primary"
    )

    if run_clicked:
        if st.session_state.extraction_count >= MAX_EXTRACTIONS_PER_SESSION:
            st.error(
                "This session has hit the demo's extraction limit. Refresh the page, "
                "or reach out for a full walkthrough with no limits."
            )
        else:
            try:
                client = Groq(api_key=api_key_input)
                with st.spinner("Extracting data with AI..."):
                    result = call_groq(client, doc_text)
                st.session_state.extraction_count += 1
                st.session_state.result = result
            except Exception:
                print(
                    "[extraction error]\n" + traceback.format_exc()
                )  # server-side log
                st.error(
                    "Couldn't process this document right now — try another file, "
                    "or let me know and I'll take a look."
                )

if st.session_state.result:
    result = st.session_state.result
    st.success("✅ Extraction complete!")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vendor", result.get("vendor_name") or "—")
    m2.metric("Invoice #", result.get("document_number") or "—")
    m3.metric("Date", result.get("date") or "—")
    total = result.get("total_amount")
    currency = result.get("currency") or ""
    m4.metric("Total", f"{currency} {total}".strip() if total is not None else "—")

    df = flatten_for_table(result)
    pd_xero_fmt = "%d/%m/%Y" if xero_date_fmt == "DD/MM/YYYY" else "%m/%d/%Y"
    xero_df = to_xero_bills_csv(
        result,
        account_code=xero_account_code,
        tax_type=xero_tax_type,
        due_days=int(xero_due_days),
        date_fmt=pd_xero_fmt,
    )
    pd_qb_fmt = "%m/%d/%Y" if qb_date_fmt == "MM/DD/YYYY" else "%d/%m/%Y"
    qb_df = to_quickbooks_csv(
        result,
        category=qb_category,
        terms=qb_terms,
        due_days=int(qb_due_days),
        date_fmt=pd_qb_fmt,
    )

    tab_data, tab_xero, tab_qb = st.tabs(
        ["📋 Extracted Data", "🟦 Xero Export", "🟩 QuickBooks Export"]
    )

    with tab_data:
        st.subheader("Table view")
        st.dataframe(df, use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="extracted_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="extracted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with st.expander("Raw JSON"):
            st.json(result)

    with tab_xero:
        st.caption(
            "Formatted for Xero's **Business > Bills to pay > Import** CSV template. "
            "Tweak the account code/tax type in the sidebar to match a client's chart of "
            "accounts — this updates instantly. Double-check both before importing."
        )
        st.dataframe(xero_df, use_container_width=True)
        xero_csv_bytes = xero_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Xero CSV (Bills import)",
            data=xero_csv_bytes,
            file_name="xero_bill_import.csv",
            mime="text/csv",
        )

    with tab_qb:
        st.caption(
            "Formatted for the column layout used by common QuickBooks Online bulk-import "
            "apps (e.g. SaasAnt Transactions, Transaction Pro Importer) — QBO has no native "
            "generic CSV bill-import screen, so one of those apps is the usual way in. "
            "Tweak the category/terms in the sidebar to match the client's chart of accounts."
        )
        st.dataframe(qb_df, use_container_width=True)
        qb_csv_bytes = qb_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download QuickBooks CSV (Bill import)",
            data=qb_csv_bytes,
            file_name="quickbooks_bill_import.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Built for invoices — happy to adapt the extracted fields, output format "
    "(Xero, QuickBooks, plain CSV), or add OCR for scanned documents to fit "
    "your specific paperwork."
)
st.caption(f"— {FREELANCER_NAME} · [{CONTACT_LINK}]({CONTACT_LINK})")
