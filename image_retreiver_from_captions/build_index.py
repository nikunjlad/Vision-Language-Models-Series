import argparse, os, json, pathlib, numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Optional captioning (enabled with --with_captions)
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
def maybe_make_captioner(use_captions: bool):
    if not use_captions:
        return None
    from transformers import BlipForConditionalGeneration, BlipProcessor
    device = "cpu"
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(device)
    model.eval()
    return (processor, model, device)

def caption_image(captioner, image: Image.Image):
    if captioner is None:
        return None
    processor, model, device = captioner
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text

def load_images(image_dir):
    SUP = {".jpg",".jpeg",".png",".bmp",".webp",".tiff"}
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUP:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths

def embed_images(model, paths, batch_size=32):
    # SentenceTransformer CLIP model can encode images directly.
    imgs = []
    results = []
    for p in tqdm(paths, desc="Embedding images"):
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except Exception as e:
            print(f"[warn] Skipping {p}: {e}")
            imgs.append(None)

    # Batch over images (keeping alignment with paths)
    batch = []
    batch_idx = []
    for i, img in enumerate(imgs):
        if img is None:  # put a zero vector placeholder
            results.append(None)
            continue
        batch.append(img)
        batch_idx.append(i)
        if len(batch) == batch_size:
            embs = model.encode(batch, batch_size=len(batch), convert_to_numpy=True, normalize_embeddings=True)
            for j, idx in enumerate(batch_idx):
                results.append((idx, embs[j]))
            batch, batch_idx = [], []

    if batch:
        embs = model.encode(batch, batch_size=len(batch), convert_to_numpy=True, normalize_embeddings=True)
        for j, idx in enumerate(batch_idx):
            results.append((idx, embs[j]))

    # Reconstruct embedding array in order
    dim = None
    arr = [None]*len(paths)
    for r in results:
        if r is None:
            continue
        idx, vec = r
        dim = vec.shape[0]
        arr[idx] = vec
    # Replace Nones with zeros
    for i in range(len(arr)):
        if arr[i] is None:
            arr[i] = np.zeros(dim, dtype="float32")
    return np.vstack(arr).astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--index_out", required=True)
    ap.add_argument("--meta_out", required=True)
    ap.add_argument("--clip_model", default="clip-ViT-L-14")
    ap.add_argument("--with_captions", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(args.clip_model, device=device)

    captioner = maybe_make_captioner(args.with_captions)

    paths = load_images(args.images_dir)
    if not paths:
        raise SystemExit("No images found.")
    print(f"Found {len(paths)} images")

    # 1) Embeddings
    vecs = embed_images(model, paths, batch_size=4)
    dim = vecs.shape[1]
    print(f"Embeddings shape: {vecs.shape}")

    # 2) FAISS index (cosine via normalized vectors + inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, args.index_out)
    print(f"Wrote index: {args.index_out}")

    # 3) Metadata
    meta = []
    for i, p in enumerate(paths):
        item = {"id": i, "path": p}
        if captioner is not None:
            try:
                cap = caption_image(captioner, Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[warn] caption failed for {p}: {e}")
                cap = None
            item["caption"] = cap
        meta.append(item)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {args.meta_out}")

if __name__ == "__main__":
    main()
