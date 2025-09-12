# Image Captioning + Vector Search (CLIP + BLIP + FAISS)

This starter kit shows how to:
1) Auto-caption images (optional) using BLIP.
2) Encode images with CLIP into a shared vision–language space.
3) Store vectors in a FAISS index (local) along with JSON metadata.
4) Query by natural language (e.g., "man in red jacket") and retrieve matching images.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Build the index
Put your images under a folder, e.g. `./sample_images/` (nested folders are fine). Then:

```bash
python build_index.py --images_dir ./sample_images                           --index_out ./index.faiss                           --meta_out ./meta.json                           --with_captions
```

Flags:
- `--with_captions`: generates captions using BLIP (`Salesforce/blip-image-captioning-large`) and stores them in metadata for display and fallback text search.
  (This is optional; retrieval itself uses CLIP embeddings.)

### 2) Query the index
```bash
python query_search.py --index ./index.faiss                            --meta ./meta.json                            --query "man in red jacket"                            --k 12
```

This prints the top results with paths and captions, and writes a small HTML gallery to `./results.html` for quick visual inspection.

## Notes & Tips
- CLIP puts text and image embeddings in the *same* space. We normalize and search with inner product (equivalent to cosine similarity).
- Captions are helpful for UI and for a second-pass text-only search, but they aren’t required for cross-modal retrieval.
- If you need a server/cluster vector DB, you can switch to Qdrant/Milvus/Weaviate. A Qdrant example is included at the bottom of `query_search.py` (commented).
- Batch process images on GPU for speed. Both BLIP and CLIP will use CUDA if available.
- Consider a re-ranker (e.g., a CLIP cross-encoder) for the top 100 hits if you need higher precision.
- For production: add metadata (timestamp, camera_id, location), deduplicate similar frames, and consider access controls.
- Safety/Privacy: if images include people, ensure your usage complies with local laws and your org’s policies.

## Troubleshooting
- If `faiss` import fails on Apple Silicon/Windows, ensure you installed the `faiss-cpu` wheel.
- Large models may require `pip install --upgrade pip` and recent `torch`.
