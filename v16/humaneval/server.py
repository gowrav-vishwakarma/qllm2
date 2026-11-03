"""Human-eval backend: the phone gets only what it taps.
  GET /                      -> index.html
  GET /api/items?s=&f=&q=&p= -> one page (30) of short summaries for section s (filter all/ok/bad, search q)
  GET /api/item?s=&i=        -> one full item
Section data lives in site/data/<id>.json and is loaded on the server only."""
import json, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
SITE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / 'site'; PER = 30; CACHE = {}

def data(s):
    if s not in CACHE:
        p = SITE / 'data' / f'{Path(s).name}.json'
        CACHE[s] = json.load(open(p)) if p.exists() else []
        CACHE[s + ':txt'] = [json.dumps(x).lower() for x in CACHE[s]]
    return CACHE[s], CACHE[s + ':txt']

def summary(x, i):
    k = []
    for f in ('name', 'mode', 'task', 'arm', 'tag'):
        if x.get(f): k.append(str(x[f]))
    if 'gold' in x and 'pred' in x: k.append(f"gold {x['gold']} · model {x['pred']}")
    if 'target' in x and 'pick' in x: k.append(f"answer “{x['target'].strip()}” · model “{x['pick'].strip()}”")
    if 'n_tokens' in x: k.append(f"{x['n_tokens']:,} tokens")
    pv = str(x.get('prompt') or x.get('text') or '')
    pv = pv[-110:] if x.get('greedy') is not None else pv[:110]
    return dict(i=i, ok=x.get('ok'), head=' · '.join(k), pv=' '.join(pv.split()))

class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200); self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ('/', '/index.html'): return self._send((SITE / 'index.html').read_bytes(), 'text/html')
        if u.path == '/api/items':
            it, txt = data(q.get('s', '')); f = q.get('f', 'all'); s = q.get('q', '').lower(); p = int(q.get('p', 0) or 0)
            idx = [i for i, x in enumerate(it) if (f == 'all' or (f == 'ok') == (x.get('ok') is True)) and (not s or s in txt[i])]
            npg = max(1, -(-len(idx) // PER)); p = min(max(p, 0), npg - 1)
            return self._send(dict(total=len(idx), page=p, pages=npg, items=[summary(it[i], i) for i in idx[p * PER:(p + 1) * PER]]))
        if u.path == '/api/item':
            it, _ = data(q.get('s', '')); i = int(q.get('i', -1))
            return self._send(it[i] if 0 <= i < len(it) else {})
        self.send_response(404); self.end_headers()

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
