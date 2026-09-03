import os, re, time, json
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests

ROOT = Path(r'd:\Nandana\MTECH\Semester 3\Projects\CP\KG-RAG-ISRO')
raw_dir = ROOT / 'data' / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)

html = requests.get('https://www.isro.gov.in', timeout=60).text
links = [urljoin('https://www.isro.gov.in', m.group(1)) for m in re.finditer(r'href=["\']([^"\']+)["\']', html, flags=re.I)]
links = sorted({
    l for l in links
    if 'isro.gov.in' in l and not any(x in l.lower() for x in ['login','search','calendar','javascript:','mailto:','tel:','sitemap','facebook','youtube','twitter','instagram','rss'])
})

# unique candidate pages; skip obvious duplicates and media files
candidates = []
seen = set()
for url in links:
    if url in seen:
        continue
    seen.add(url)
    parsed = urlparse(url)
    if parsed.path.lower().endswith(('.pdf','.jpg','.png','.jpeg','.gif','.mp4','.zip','.mp3','.xml')):
        continue
    candidates.append(url)

saved = 0
for idx, url in enumerate(candidates[:160], 1):
    # avoid API abuse: throttle and retry on 429
    if idx > 1:
        time.sleep(2.5)

    safe = re.sub(r'[^a-zA-Z0-9]+', '_', url.lower()).strip('_') or 'page'
    target = raw_dir / f'{safe}.md'
    if target.exists():
        print(f'SKIP_EXIST {idx}/{len(candidates[:160])} {url}')
        continue

    last_error = None
    for attempt in range(3):
        try:
            payload = {'url': url, 'formats': ['markdown'], 'onlyMainContent': True}
            headers = {'Authorization': 'Bearer ' + os.environ['FIRECRAWL_API_KEY'], 'Content-Type': 'application/json'}
            r = requests.post('https://api.firecrawl.dev/v1/scrape', json=payload, headers=headers, timeout=90)
            if r.status_code == 429:
                time.sleep(8)
                last_error = '429'
                continue
            if r.status_code != 200:
                last_error = f'HTTP {r.status_code}'
                print(f'SKIP_STATUS {url} {r.status_code} {r.text[:200]}')
                break
            data = r.json()
            md = (data.get('data') or {}).get('markdown') or ''
            if not md.strip():
                print(f'EMPTY {url}')
                break
            target.write_text(f'# Source: {url}\n\n{md.strip()}\n', encoding='utf-8')
            saved += 1
            print(f'{idx}/{len(candidates[:160])} SAVED {url}')
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(5)
    else:
        print(f'FAILED {url} {last_error}')

print('TOTAL_SAVED', saved)
print('RAW_COUNT=', len(list(raw_dir.glob('*.md'))))
