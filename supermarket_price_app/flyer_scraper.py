from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class FlyerItem:
    store: str
    title: str
    price_text: str
    source_url: str


STORE_FLYER_URLS = {
    "Walmart": "https://www.walmart.ca/en/flyer",
    "Save-On-Foods": "https://www.saveonfoods.com/flyer",
    "Thrifty Foods": "https://www.thriftyfoods.com/flyer",
}


def fetch_flyer_html(url: str) -> str:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.text


def extract_candidate_items(store: str, html: str, source_url: str) -> list[FlyerItem]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for text_node in soup.stripped_strings:
        text = text_node.strip()
        if "$" not in text:
            continue
        if len(text) > 120:
            continue
        candidates.append(
            FlyerItem(
                store=store,
                title=text,
                price_text=text,
                source_url=source_url,
            )
        )
        if len(candidates) >= 40:
            break

    return candidates


def scrape_flyers(stores: Iterable[str] | None = None) -> list[FlyerItem]:
    results: list[FlyerItem] = []
    selected = STORE_FLYER_URLS.keys() if stores is None else stores

    for store in selected:
        url = STORE_FLYER_URLS.get(store)
        if not url:
            continue
        html = fetch_flyer_html(url)
        results.extend(extract_candidate_items(store, html, url))

    return results


def main() -> None:
    items = scrape_flyers()
    for item in items:
        print(f"{item.store}: {item.title} ({item.source_url})")


if __name__ == "__main__":
    main()
