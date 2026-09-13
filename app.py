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


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="AI Document Data Extractor", page_icon="📄")

st.title("📄 AI Document Data Extractor")
st.caption(
    "Upload a PDF and let AI pull out structured data — no manual typing, "
    "and export straight to a Xero-ready bill import. Powered by Groq."
)

if "extraction_count" not in st.session_state:
    st.session_state.extraction_count = 0
if "result" not in st.session_state:
    st.session_state.result = None

shared_api_key = get_shared_api_key()

with st.sidebar:
    st.header("Setup")
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

    st.header("Xero export settings")
    st.caption(
        "Used to build the Xero-ready CSV — match these to the client's Xero org."
    )
    xero_account_code = st.text_input("Default account code", value="400")
    xero_tax_type = st.text_input(
        "Default tax type (exact name from client's Xero)", value="Tax Exempt"
    )
    xero_due_days = st.number_input(
        "Due date = invoice date +", min_value=0, max_value=120, value=30, step=1
    )
    xero_date_fmt = st.selectbox("Date format", ["DD/MM/YYYY", "MM/DD/YYYY"], index=0)

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

    st.success("Extraction complete!")
    st.subheader("Extracted JSON")
    st.json(result)

    df = flatten_for_table(result)
    st.subheader("Table view")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="extracted_data.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    st.download_button(
        "⬇️ Download Excel",
        data=excel_buffer.getvalue(),
        file_name="extracted_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Xero-ready export")
    pd_date_fmt = "%d/%m/%Y" if xero_date_fmt == "DD/MM/YYYY" else "%m/%d/%Y"
    xero_df = to_xero_bills_csv(
        result,
        account_code=xero_account_code,
        tax_type=xero_tax_type,
        due_days=int(xero_due_days),
        date_fmt=pd_date_fmt,
    )
    st.caption(
        "Formatted for Xero's **Business > Bills to pay > Import** CSV template. "
        "Tweak the account code/tax type in the sidebar to match a client's chart of accounts — "
        "the table below updates instantly. Double-check both before importing."
    )
    st.dataframe(xero_df, use_container_width=True)

    xero_csv_bytes = xero_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Xero CSV (Bills import)",
        data=xero_csv_bytes,
        file_name="xero_bill_import.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Built for invoices — happy to adapt the extracted fields, output format "
    "(Xero, QuickBooks, plain CSV), or add OCR for scanned documents to fit "
    "your specific paperwork."
)
