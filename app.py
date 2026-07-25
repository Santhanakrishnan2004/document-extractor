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

import pandas as pd
import pdfplumber
import streamlit as st
from groq import Groq

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

MODEL = "llama-3.3-70b-versatile"  # fast + free-tier friendly on Groq

# ---------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="AI Document Data Extractor", page_icon="📄")

st.title("📄 AI Document Data Extractor")
st.caption(
    "Upload a PDF and let AI pull out structured data — no manual typing. "
    "Powered by Groq (Llama 3.3 70B)."
)

api_key = os.environ.get("GROQ_API_KEY", "")
with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Groq API Key",
        value=api_key,
        type="password",
        help="Get a free key at console.groq.com",
    )
    st.markdown("[Get a free Groq API key →](https://console.groq.com/keys)")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file and not api_key_input:
    st.warning("Add your Groq API key in the sidebar to run extraction.")

if uploaded_file and api_key_input:
    with st.spinner("Reading PDF..."):
        doc_text = extract_text_from_pdf(uploaded_file)

    if not doc_text.strip():
        st.error("Couldn't extract any text from this PDF. It might be a scanned image — OCR support can be added.")
    else:
        with st.expander("Show extracted raw text"):
            st.text(doc_text[:3000])

        if st.button("Extract structured data", type="primary"):
            try:
                client = Groq(api_key=api_key_input)
                with st.spinner("Extracting data with AI..."):
                    result = call_groq(client, doc_text)

                st.success("Extraction complete!")
                st.subheader("Extracted JSON")
                st.json(result)

                df = flatten_for_table(result)
                st.subheader("Table view")
                st.dataframe(df, use_container_width=True)

                # Download buttons
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

            except Exception as e:
                st.error(f"Extraction failed: {e}")

st.markdown("---")
st.caption(
    "This is a demo build. For client work: swap the schema above for whatever "
    "fields the client needs, add OCR for scanned docs, and wrap this logic "
    "into whatever delivery format they want (API, script, or this same UI)."
)
