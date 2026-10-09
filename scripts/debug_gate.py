import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clerk import privilege, store
import clerk.retrieval as R

q = "Which customer records contain social security numbers and account numbers in the support ticket export?"
chunks = R.retrieve(q, k=6)
for ch in chunks:
    meta = store.doc_meta(ch["doc_id"]) or {}
    need, prior = privilege.stage1_screen(ch["text"])
    sus = privilege._suspects(ch["text"])
    zs = privilege.stage1_5_zero_shot(ch["text"], sus) if need else None
    print(ch["doc_id"][:8], repr(meta.get("title", "")[:35]),
          "stage1:", need, round(prior, 2), "sus:", sus, "zs:", zs)
