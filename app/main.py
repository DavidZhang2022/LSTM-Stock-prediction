from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


@dataclass(frozen=True)
class ProductPrice:
    product_id: str
    name: str
    brand: str
    category: str
    store: str
    price: float
    unit: str
    flyer_source: str


PRODUCTS: list[ProductPrice] = [
    ProductPrice(
        product_id="milk-2l",
        name="2% Milk 2L",
        brand="Dairyland",
        category="Dairy",
        store="Walmart",
        price=5.49,
        unit="each",
        flyer_source="https://www.walmart.ca/en/flyer",
    ),
    ProductPrice(
        product_id="milk-2l",
        name="2% Milk 2L",
        brand="Dairyland",
        category="Dairy",
        store="Save-On-Foods",
        price=5.99,
        unit="each",
        flyer_source="https://www.saveonfoods.com/flyer",
    ),
    ProductPrice(
        product_id="milk-2l",
        name="2% Milk 2L",
        brand="Island Farms",
        category="Dairy",
        store="Thrifty Foods",
        price=6.29,
        unit="each",
        flyer_source="https://www.thriftyfoods.com/flyer",
    ),
    ProductPrice(
        product_id="eggs-12",
        name="Large Eggs 12-pack",
        brand="Burnbrae",
        category="Dairy",
        store="Walmart",
        price=4.79,
        unit="pack",
        flyer_source="https://www.walmart.ca/en/flyer",
    ),
    ProductPrice(
        product_id="eggs-12",
        name="Large Eggs 12-pack",
        brand="Burnbrae",
        category="Dairy",
        store="Save-On-Foods",
        price=5.29,
        unit="pack",
        flyer_source="https://www.saveonfoods.com/flyer",
    ),
    ProductPrice(
        product_id="eggs-12",
        name="Large Eggs 12-pack",
        brand="Sunrise",
        category="Dairy",
        store="Thrifty Foods",
        price=5.49,
        unit="pack",
        flyer_source="https://www.thriftyfoods.com/flyer",
    ),
    ProductPrice(
        product_id="bread-675g",
        name="Whole Wheat Bread 675g",
        brand="Dempster's",
        category="Bakery",
        store="Walmart",
        price=3.49,
        unit="loaf",
        flyer_source="https://www.walmart.ca/en/flyer",
    ),
    ProductPrice(
        product_id="bread-675g",
        name="Whole Wheat Bread 675g",
        brand="Dempster's",
        category="Bakery",
        store="Save-On-Foods",
        price=3.99,
        unit="loaf",
        flyer_source="https://www.saveonfoods.com/flyer",
    ),
    ProductPrice(
        product_id="bread-675g",
        name="Whole Wheat Bread 675g",
        brand="Country Harvest",
        category="Bakery",
        store="Thrifty Foods",
        price=4.29,
        unit="loaf",
        flyer_source="https://www.thriftyfoods.com/flyer",
    ),
]

DEALS = [
    {
        "store": "Walmart",
        "title": "Fresh produce bundle",
        "description": "Apples, bananas, and salad mix combo for $9.99.",
        "valid_until": "2025-01-15",
    },
    {
        "store": "Save-On-Foods",
        "title": "BC salmon fillets",
        "description": "Fresh salmon fillets $12.99/lb.",
        "valid_until": "2025-01-12",
    },
    {
        "store": "Thrifty Foods",
        "title": "Family pasta night",
        "description": "Pasta + sauce bundle for $6.49.",
        "valid_until": "2025-01-14",
    },
]

app = FastAPI(title="Victoria Grocery Price Compare")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Victoria Price Compare</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; }
            header { margin-bottom: 16px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
            .price { font-weight: bold; }
            .tag { background: #eef; border-radius: 999px; padding: 4px 8px; font-size: 12px; }
            input { padding: 8px; width: 100%; margin: 12px 0; }
            table { width: 100%; border-collapse: collapse; margin-top: 12px; }
            th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; }
        </style>
    </head>
    <body>
        <header>
            <h1>Victoria Grocery Price Compare</h1>
            <p>Compare Walmart, Save-On-Foods, and Thrifty Foods in Victoria, BC.</p>
        </header>
        <section class=\"card\">
            <h2>Search products</h2>
            <input id=\"search\" placeholder=\"Search by product, brand, or category\" />
            <div id=\"results\"></div>
        </section>
        <section>
            <h2>Weekly deals</h2>
            <div id=\"deals\" class=\"grid\"></div>
        </section>
        <script>
            async function fetchProducts(query) {
                const params = new URLSearchParams();
                if (query) params.set('q', query);
                const response = await fetch(`/api/products?${params}`);
                return response.json();
            }

            async function fetchDeals() {
                const response = await fetch('/api/deals');
                return response.json();
            }

            function renderTable(items) {
                if (!items.length) {
                    return '<p>No matches yet. Try another search.</p>';
                }
                const rows = items.map(item => `
                    <tr>
                        <td>${item.name}</td>
                        <td>${item.brand}</td>
                        <td>${item.category}</td>
                        <td>${item.store}</td>
                        <td class=\"price\">$${item.price.toFixed(2)} / ${item.unit}</td>
                        <td><a href=\"${item.flyer_source}\" target=\"_blank\" rel=\"noopener\">Flyer</a></td>
                    </tr>
                `).join('');
                return `
                    <table>
                        <thead>
                            <tr>
                                <th>Product</th>
                                <th>Brand</th>
                                <th>Category</th>
                                <th>Store</th>
                                <th>Price</th>
                                <th>Flyer</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                `;
            }

            function renderDeals(items) {
                const cards = items.map(item => `
                    <div class=\"card\">
                        <span class=\"tag\">${item.store}</span>
                        <h3>${item.title}</h3>
                        <p>${item.description}</p>
                        <p><strong>Valid until:</strong> ${item.valid_until}</p>
                    </div>
                `).join('');
                document.getElementById('deals').innerHTML = cards;
            }

            async function handleSearch(event) {
                const query = event.target.value.trim();
                const items = await fetchProducts(query);
                document.getElementById('results').innerHTML = renderTable(items);
            }

            document.getElementById('search').addEventListener('input', handleSearch);

            fetchProducts('').then(items => {
                document.getElementById('results').innerHTML = renderTable(items);
            });
            fetchDeals().then(renderDeals);
        </script>
    </body>
    </html>
    """


@app.get("/api/products")
def list_products(q: str | None = Query(default=None, description="Search term")) -> list[dict[str, str | float]]:
    matched = filter_products(PRODUCTS, q)
    return [
        {
            "product_id": item.product_id,
            "name": item.name,
            "brand": item.brand,
            "category": item.category,
            "store": item.store,
            "price": item.price,
            "unit": item.unit,
            "flyer_source": item.flyer_source,
        }
        for item in matched
    ]


@app.get("/api/deals")
def list_deals() -> list[dict[str, str]]:
    return DEALS


def filter_products(items: Iterable[ProductPrice], query: str | None) -> list[ProductPrice]:
    if not query:
        return list(items)
    lowered = query.lower()
    return [
        item
        for item in items
        if lowered in item.name.lower()
        or lowered in item.brand.lower()
        or lowered in item.category.lower()
        or lowered in item.store.lower()
    ]
