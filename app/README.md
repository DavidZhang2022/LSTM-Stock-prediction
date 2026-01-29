# Victoria Grocery Price Compare (MVP)

This is a small FastAPI app that lets you compare sample prices across Walmart, Save-On-Foods, and Thrifty Foods in Victoria, BC.

## Run locally

```bash
cd app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

## Notes
- The product list is hard-coded for the MVP. Replace with a data pipeline or database when ready.
- Deals are sample entries for the weekly deals carousel.
