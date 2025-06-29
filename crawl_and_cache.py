import os
import shelve
import asyncio
import glob
import pickle
import gzip
from urllib.parse import urljoin, urlparse

import numpy as np
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
MAGENTO_BASE_URL     = os.getenv("MAGENTO_BASE_URL", "").rstrip('/')
MAGENTO_STORE_CODE   = os.getenv("MAGENTO_STORE_CODE", "")
MAGENTO_BEARER_TOKEN = os.getenv("MAGENTO_BEARER_TOKEN", "")
DEFAULT_MAX_PAGES    = int(os.getenv("DEFAULT_MAX_PAGES", "200"))
CRAWL_URL            = os.getenv("CRAWL_URL", "")
CACHE_DIR            = os.getenv("CACHE_DIR", ".new_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DB             = os.getenv("CACHE_DB", "cache")
CACHE_PATH           = os.path.join(CACHE_DIR, CACHE_DB)

client = OpenAI(api_key=OPENAI_API_KEY)

MAGENTO_HEADERS = {
    "Authorization": f"Bearer {MAGENTO_BEARER_TOKEN}",
    "Content-Type": "application/json"
}

async def fetch_url(session, url):
    try:
        r = await session.get(url, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            soup = BeautifulSoup(r.text, 'html.parser')
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            return url, soup.get_text(separator=' ', strip=True), soup
    except Exception:
        pass
    return url, None, None

async def crawl_website_async(start_url, max_pages=DEFAULT_MAX_PAGES):
    visited, to_visit, texts = set(), [start_url], []
    base = "{0.scheme}://{0.netloc}".format(urlparse(start_url))
    async with httpx.AsyncClient(follow_redirects=True) as session:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            fetched, text, soup = await fetch_url(session, url)
            if text:
                texts.append(text)
                visited.add(fetched)
                for link in soup.find_all("a", href=True):
                    href = urljoin(base, link['href']).split('#')[0]
                    if base in href and href not in visited and href not in to_visit:
                        to_visit.append(href)
    return "\n\n".join(texts)

async def fetch_magento_products(page_size=100):
    items = []
    async with httpx.AsyncClient() as session:
        page = 1
        while True:
            url = f"{MAGENTO_BASE_URL}/rest/{MAGENTO_STORE_CODE}/V1/products"
            params = {"searchCriteria[currentPage]": page, "searchCriteria[pageSize]": page_size}
            resp = await session.get(url, headers=MAGENTO_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json().get("items", [])
            if not batch:
                break
            items.extend(batch)
            page += 1
    return items

def product_to_text(product: dict) -> str:
    out = [
        f"SKU: {product.get('sku')}",
        f"Name: {product.get('name')}",
        f"Price: {product.get('price')}",
        f"Status: {'Enabled' if product.get('status') == 1 else 'Disabled'}"
    ]
    for attr in product.get("custom_attributes", []):
        out.append(f"{attr.get('attribute_code')}: {attr.get('value')}")
    return "\n".join(out)

def load_pdfs(folder="extra_info"):
    docs = []
    if not os.path.isdir(folder): return docs
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            if pages:
                docs.append("\n".join(pages))
        except Exception as e:
            print(f"PDF load error {path}: {e}")
    return docs

def split_text(text, max_tokens=300):
    words = text.split()
    return [' '.join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks, batch_size=100):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(model="text-embedding-3-small", input=chunks[i:i+batch_size])
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings).astype("float32")

def save_compressed(db, key, obj):
    """Save a large object to shelve using pickle+gzip compression."""
    db[key] = gzip.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

def load_compressed(db, key):
    """Load a compressed object from shelve."""
    return pickle.loads(gzip.decompress(db[key]))

def main():
    with shelve.open(CACHE_PATH) as db:
        # Crawl website
        if CRAWL_URL:
            print(f"Crawling website: {CRAWL_URL}")
            crawled = asyncio.run(crawl_website_async(CRAWL_URL, DEFAULT_MAX_PAGES))
            crawl_chunks = split_text(crawled)
            crawl_embs = embed_chunks(crawl_chunks)
            crawl_key = f"crawl||{CRAWL_URL}||{DEFAULT_MAX_PAGES}"
            save_compressed(db, crawl_key, (crawl_chunks, crawl_embs))
            print(f"Saved crawl data under key: {crawl_key}")

        # Magento products
        mag_key = f"magento||{MAGENTO_STORE_CODE}"
        print("Fetching Magento products...")
        prods = asyncio.run(fetch_magento_products(page_size=200))
        texts = [product_to_text(p) for p in prods] + load_pdfs()
        mag_chunks = []
        for t in texts:
            mag_chunks.extend(split_text(t))
        mag_embs = embed_chunks(mag_chunks)
        save_compressed(db, mag_key, (mag_chunks, mag_embs))
        print(f"Saved Magento data under key: {mag_key}")

if __name__ == "__main__":
    main()
