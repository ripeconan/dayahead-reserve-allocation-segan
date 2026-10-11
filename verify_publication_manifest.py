"""Verify published scientific files using Python standard library only."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/'PUBLICATION_MANIFEST.json').read_text(encoding='utf8'))
for e in m['files']:
 p=r/e['path'];b=p.read_bytes()
 assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],e['path']
print('PASS:',len(m['files']),'published file identities verified')
