import sys
sys.path.append("backend")
import requests
import json
from app.ingestion.epo_auth import EPOAuthManager

mgr = EPOAuthManager()
token = mgr.get_valid_token()

doc_key = "epodoc/EP4822722"
url = f"https://ops.epo.org/3.2/rest-services/published-data/publication/{doc_key}/claims"
headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

res = requests.get(url, headers=headers)
print("Claims fetch status:", res.status_code)
if res.status_code == 200:
    data = res.json()
    print("Claims JSON:")
    print(json.dumps(data, indent=2)[:1000])
else:
    print("Claims Error:", res.text[:200])
