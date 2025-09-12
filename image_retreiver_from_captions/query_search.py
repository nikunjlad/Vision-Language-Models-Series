import argparse, json, numpy as np, webbrowser, html
from sentence_transformers import SentenceTransformer
import faiss, os

def load_meta(path):
    with open(path, "r") as f:
        return json.load(f)

def to_html(results, out_path="results.html"):
    rows = []
    rows.append("<h2>Search Results</h2>")
    rows.append("<style>img{max-height:180px;margin:6px;border-radius:8px} .card{display:inline-block;padding:8px;margin:8px;border:1px solid #ddd;border-radius:12px;vertical-align:top}</style>")
    for r in results:
        p = html.escape(r["path"])
        cap = html.escape(r.get("caption") or "")
        score = f'{r["score"]:.3f}'
        card = f'<div class="card"><div><img src="{p}" /></div><div><b>score:</b> {score}</div><div>{cap}</div><div style="max-width:320px;word-wrap:break-word;font-size:12px;color:#666">{p}</div></div>'
        rows.append(card)
    with open(out_path, "w") as f:
        f.write("\n".join(rows))
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--clip_model", default="clip-ViT-L-14")
    args = ap.parse_args()

    meta = load_meta(args.meta)
    dim = None
    index = faiss.read_index(args.index)

    model = SentenceTransformer(args.clip_model)
    qvec = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(qvec, args.k)
    ids = I[0].tolist()
    scores = D[0].tolist()

    results = []
    for rank, (idx, sc) in enumerate(zip(ids, scores), 1):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx].copy()
        item["score"] = float(sc)
        item["rank"] = rank
        results.append(item)

    # Pretty print & HTML
    for r in results:
        print(f'#{r["rank"]:02d} score={r["score"]:.3f} path="{r["path"]}" caption="{r.get("caption")}"')
    out = to_html(results, out_path="results.html")
    print(f"Saved {out}. Open it in a browser to preview.")

    # --- Qdrant example (optional) ---
    # If you want to push to Qdrant instead of FAISS, see this sketch:
    # from qdrant_client import QdrantClient
    # from qdrant_client.models import VectorParams, Distance, PointStruct
    #
    # client = QdrantClient(url="http://localhost:6333")
    # client.recreate_collection(
    #     collection_name="images",
    #     vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    # )
    # points = [PointStruct(id=i, vector=vecs[i].tolist(), payload=meta[i]) for i in range(len(meta))]
    # client.upsert(collection_name="images", points=points)
    #
    # Search:
    # hits = client.search(collection_name="images", query_vector=qvec[0].tolist(), limit=args.k)
    # results = [{"path": h.payload["path"], "caption": h.payload.get("caption"), "score": h.score} for h in hits]

if __name__ == "__main__":
    main()
