# Victoria Supermarket Price Comparison App (MVP Plan)

## Goal
Build a local price comparison app focused on Victoria, BC supermarkets, initially covering:
- Walmart
- Save-On-Foods
- Thrifty Foods

The app helps users quickly compare prices across these stores, track deals, and build a shopping list optimized for cost and convenience.

## Target Users
- Students and families seeking the lowest grocery prices.
- Budget-conscious shoppers looking for weekly deals.
- New residents looking to understand local pricing.

## MVP Scope
### Core Features
1. **Product Search & Comparison**
   - Search by product name, brand, or category.
   - Show price comparisons across Walmart, Save-On-Foods, Thrifty Foods.

2. **Weekly Deals Feed**
   - Highlight flyer deals and in-store promotions.
   - Filter by store and category.

3. **Shopping List Optimizer**
   - Add items to a list and compare total cost across stores.
   - Basic best-store suggestion for the list.

4. **Store Info**
   - Store locations, hours, and basic info for Victoria.

### Out of Scope (Later Phases)
- User accounts and personalization.
- Complex routing or multi-store optimization.
- Barcode scanning.
- Cross-region expansion.

## Data Strategy
### Sources (Initial)
- Public flyers and weekly deals from store websites.
- Manually curated price lists for pilot testing.

### Potential Improvements
- Semi-automated scraping (if allowed by store terms).
- Partner APIs (if available).
- Crowd-sourced price reporting (later phase).

## UX Flow
1. User opens app → sees search bar + deals carousel.
2. Search for an item → comparison table by store.
3. Add items to list → compare total list cost.
4. Browse deals → add to list.

## Technical Stack (Suggested)
### Frontend
- React Native (cross-platform iOS/Android).
- Optional PWA for quick web access.

### Backend
- Node.js + Express or FastAPI.
- PostgreSQL for product/price data.

### Data Pipeline
- Scheduled jobs to refresh deals.
- Admin interface to validate imported prices.

## MVP Milestones
1. **Week 1–2**: Requirements + UI wireframes.
2. **Week 3–4**: Basic search and comparison UI with mock data.
3. **Week 5–6**: Data ingestion pipeline (manual or semi-automated).
4. **Week 7–8**: MVP release for local testing.

## Risks & Mitigations
- **Data accuracy**: Start with curated datasets and manual checks.
- **Store terms of use**: Verify scraping policies before automating.
- **Coverage gaps**: Focus on staple products for MVP.

## Next Steps
- Validate MVP features with 5–10 target users in Victoria.
- Build a small dataset of 200–500 common products.
- Prototype the comparison UI and test usability.
