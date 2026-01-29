# Victoria Grocery Price Compare (MVP)

This is a small FastAPI app that lets you compare sample prices across Walmart, Save-On-Foods, and Thrifty Foods in Victoria, BC, including links back to the current store flyers.

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
- A future step is ingesting flyer data into a structured table (store, item, price, unit, valid dates, source URL) before loading it into the app.

## Flyer scraping (prototype)
The app now includes a prototype scraper in `app/flyer_scraper.py` that fetches flyer pages and extracts price-like text snippets. Before using it in production, verify each store's terms of service and robots.txt policies.

```bash
python flyer_scraper.py
```
