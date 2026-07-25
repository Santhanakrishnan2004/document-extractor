# AI Document Data Extractor

Upload a PDF (invoice, receipt, resume, contract, etc.) → AI extracts structured
fields → download as CSV/Excel. Built with Streamlit + Groq (free, fast LLM API).

This is meant as a **portfolio/demo piece** for freelance gigs (Fiverr etc.)
selling "AI document extraction" or "AI agent" services.

---

## 1. Get a free Groq API key

1. Go to https://console.groq.com/keys
2. Sign up (free) and create an API key
3. Copy it — you'll paste it into the app's sidebar, or set it as an
   environment variable (see below)

Groq's free tier has generous rate limits and is plenty for demos and
light client usage.

## 2. Run it locally

```bash
# From this folder
pip install -r requirements.txt

# Option A: set the key as an env var
export GROQ_API_KEY="your_key_here"      # Mac/Linux
# or on Windows PowerShell:
# $env:GROQ_API_KEY="your_key_here"

streamlit run app.py
```

The app opens in your browser. If you didn't set the env var, just paste
your key into the sidebar field.

## 3. Test it

Grab any sample invoice/receipt PDF (search "sample invoice pdf" or make
one in Word/Google Docs and export as PDF). Upload it, click
"Extract structured data", and you should see clean JSON + a table +
download buttons.

## 4. Customize for your niche

Open `app.py` and edit two things near the top:

- `DOCUMENT_TYPE` — e.g. `"invoice"`, `"receipt"`, `"resume"`, `"purchase order"`
- `EXTRACTION_SCHEMA` — the exact fields you want pulled out. This is the
  main thing you'll change per client — every client wants slightly
  different fields.

## 5. Deploy it so you have a live demo link (not just code)

Easiest free option: **Streamlit Community Cloud**
1. Push this folder to a public GitHub repo
2. Go to https://share.streamlit.io, connect your GitHub, deploy `app.py`
3. Add `GROQ_API_KEY` as a "secret" in the app settings (not hardcoded)
4. You get a public URL — use this as your demo link in Fiverr/LinkedIn

## 6. Record a demo

Screen-record a 30-60 second clip: upload a sample invoice → click extract
→ show the clean table/download. This GIF/video is your strongest selling
asset — put it as the first image on your Fiverr gig.

## 7. Positioning for Fiverr

- Gig title: "I will build an AI agent to extract data from your documents or PDFs"
- Lead with the demo video/GIF
- Offer 3 tiers: basic (fixed schema, few docs) → standard (custom schema,
  batch) → premium (chatbot/interface wrapper, ongoing automation)
- Tags: AI agent, data extraction, automation, AI chatbot, document processing

## Known limitations (be upfront with clients about these, or fix them)

- Only handles text-based PDFs right now — scanned/image PDFs need OCR
  (can add `pytesseract` + `pdf2image` for that)
- One document at a time — batch processing is a good "premium tier" upsell
- No error correction/validation on extracted numbers — worth a sanity-check
  step for financial data
