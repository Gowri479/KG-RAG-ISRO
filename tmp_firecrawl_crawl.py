import os, re, json
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
    if 'isro.gov.in' in l and not any(x in l.lower() for x in ['login','search','calendar','javascript:','mailto:','tel:','sitemap','facebook','youtube','twitter','instagram'])
})
selected = []
seen = set()
for url in links[:80]:
    if url in seen:
        continue
    seen.add(url)
    parsed = urlparse(url)
    if parsed.path.lower().endswith(('.pdf','.jpg','.png','.jpeg','.gif','.mp4','.zip')):
        continue
    selected.append(url)

for idx, url in enumerate(selected, 1):
    try:
        payload = {'url': url, 'formats': ['markdown'], 'onlyMainContent': True}
        headers = {'Authorization': 'Bearer ' + os.environ['FIRECRAWL_API_KEY'], 'Content-Type': 'application/json'}
        r = requests.post('https://api.firecrawl.dev/v1/scrape', json=payload, headers=headers, timeout=90)
        if r.status_code != 200:
            print(f'SKIP {url} {r.status_code}')
            continue
        data = r.json()
        md = (data.get('data') or {}).get('markdown') or ''
        if not md.strip():
            print(f'EMPTY {url}')
            continue
        safe = re.sub(r'[^a-zA-Z0-9]+', '_', url.lower()).strip('_') or 'page'
        file_path = raw_dir / f'{safe}.md'
        file_path.write_text(f'# Source: {url}\n\n{md.strip()}\n', encoding='utf-8')
        print(f'{idx}/{len(selected)} SAVED {url}')
    except Exception as exc:
        print(f'ERROR {url} {exc}')

print('RAW_COUNT=' + str(len(list(raw_dir.glob('*.md')))))
