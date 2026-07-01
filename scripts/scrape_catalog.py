import argparse
import json
import re
import sys
import requests
from bs4 import BeautifulSoup

def slugify(text):
    text = re.sub(r'[^a-zA-Z0-9]', '-', text)
    text = text.lower()
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

def map_test_type_by_name(name):
    name_l = name.lower()
    if "simulation" in name_l:
        return ["S"]
    if "automata" in name_l:
        return ["S"]
    if any(k in name_l for k in ["personality", "behavior", "opq", "work styles", "motivation", "behavioral"]):
        return ["P"]
    if any(k in name_l for k in ["aptitude", "ability", "reasoning", "cognitive", "calculation", "checking", "comprehension", "verify", "numerical", "inductive", "deductive", "spatial", "mechanical"]):
        return ["A"]
    if any(k in name_l for k in ["situational", "judgement", "sjt", "biodata"]):
        return ["B"]
    if "competenc" in name_l:
        return ["C"]
    if any(k in name_l for k in ["360", "feedback", "coach"]):
        return ["D"]
    if any(k in name_l for k in ["exercise", "role play", "in-tray", "group exercise"]):
        return ["E"]
    return ["K"]

def crawl():
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    url = "https://online.shl.com/products?producttypes=1"
    print(f"Fetching catalog from {url}...", file=sys.stderr)
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="myTable")
    if not table:
        print("Error: Could not find table #myTable on the page.", file=sys.stderr)
        return []
        
    rows = table.find_all("tr")[1:]
    items = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        name = cells[1].get_text(strip=True)
        description = cells[2].get_text(strip=True)
        
        # Parse languages
        langs_raw = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        languages = sorted(list(set(l.strip() for l in langs_raw.split(",") if l.strip())))
        
        # Parse job levels
        levels_raw = cells[4].get_text(strip=True) if len(cells) > 4 else ""
        job_levels = sorted(list(set(l.strip() for l in levels_raw.split(",") if l.strip())))
        
        slug = slugify(name)
        url_item = f"https://www.shl.com/products/product-catalog/view/{slug}/"
        test_type = map_test_type_by_name(name)
        
        items.append({
            "name": name,
            "url": url_item,
            "test_type": test_type,
            "description": description,
            "job_levels": job_levels,
            "languages": languages,
            "remote_testing": True,
            "adaptive_irt": "adaptive" in description.lower()
        })
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/data/catalog.json")
    ap.add_argument("--max-pages", type=int, default=32, help="listing pages to crawl (ignored in new version)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (ignored)")
    ap.add_argument("--skip-details", action="store_true", help="skip per-item detail page fetch (ignored)")
    args = ap.parse_args()

    items = crawl()

    try:
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing seed items from {args.out}", file=sys.stderr)
    except Exception:
        existing = []

    seen_names = {item["name"].lower().strip() for item in existing}
    seen_urls = {item["url"].lower().strip() for item in existing}

    merged = list(existing)
    new_added = 0
    for item in items:
        n = item["name"].lower().strip()
        u = item["url"].lower().strip()
        if n not in seen_names and u not in seen_urls:
            merged.append(item)
            seen_names.add(n)
            seen_urls.add(u)
            new_added += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(merged)} items (added {new_added} new ones) to {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()

