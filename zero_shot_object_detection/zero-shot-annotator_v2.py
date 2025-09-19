# -*- coding: utf-8 -*-
"""
Zero-shot pre-annotation with OWLv2 -> YOLO labels (+ CVAT-friendly)
- Processor-based resize (OWLv2); boxes mapped to ORIGINAL image size
- YOLO .txt per image (CVAT 'YOLO 1.1' import)
- Visualization images
- manifest.json tracks which files are already annotated (idempotent, incremental)
- Preserves relative subfolder structure in outputs
"""
import os, sys, json, time, random, shutil, hashlib, tempfile
from pathlib import Path
from typing import List, Tuple
from contextlib import nullcontext

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Owlv2ForObjectDetection

# -------------------
# CONFIG
# -------------------
MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical YOLO classes you want to train on
YOLO_CLASSES = ["person", "vehicle", "street_sign", "weapon"]

# Zero-shot prompts (keep concise for bootstrapping)
ZS_LABELS = [
    "man", "woman", "soldier",
    "vehicle",
    "weapon", "knife", "shotgun", "rifle", "pistol", "gun",
    "street_sign",
]

# Map synonyms/phrases → canonical training classes (skip those that don't map)
ALIASES = {
    "man": "person",
    "woman": "person",
    "soldier": "person",
    "vehicle": "vehicle",
    "street_sign": "street_sign",
    "weapon": "weapon",
    "gun": "weapon",
    "pistol": "weapon",
    "rifle": "weapon",
    "shotgun": "weapon",
    "knife": "weapon",
}

SCORE_THRESHOLD = 0.30          # detector score filter
LABEL_CHUNK = 24                # chunk label prompts to lower VRAM use

# I/O
SRC_DIR  = Path("/home/nikunj/research/Vision-Language-Models-Series/data/zero-shot-test-images")
OUT_ROOT = Path("/home/nikunj/research/Vision-Language-Models-Series/data/zero-shot-test-images/zero_shot_annotated")
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"
OUT_VIS    = OUT_ROOT / "viz"
for p in (OUT_IMAGES, OUT_LABELS, OUT_VIS): p.mkdir(parents=True, exist_ok=True)

# Manifest settings
MANIFEST_PATH = OUT_ROOT / "manifest.json"
FORCE_REPROCESS = False      # set True to force re-annotate everything
USE_HASH = False             # True = content hash; False = size+mtime (faster)

# Optional: quick train/val split (set to 0 to skip)
VAL_FRACTION = 0.0  # e.g., 0.1

# -------------------
# UTILITIES
# -------------------
def clamp_xyxy(b: torch.Tensor, w: int, h: int) -> torch.Tensor:
    b[:, 0::2] = b[:, 0::2].clamp(0, w - 1)
    b[:, 1::2] = b[:, 1::2].clamp(0, h - 1)
    return b

def to_yolo_line(xyxy: Tuple[float,float,float,float], cls_id: int, W: int, H: int) -> str:
    x1, y1, x2, y2 = xyxy
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    if w <= 0 or h <= 0:
        return ""
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return (r - l, b - t)
    if hasattr(font, "getsize"):
        return font.getsize(text)
    return (max(1, 7 * len(text)), 12)

def draw_boxes(img, boxes, labels, scores, min_score=0.0):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    palette = {"person":"red","vehicle":"blue","street_sign":"yellow","weapon":"green","animal":"purple"}
    for (x1,y1,x2,y2), lab, sc in zip(boxes, labels, scores):
        if sc < min_score: continue
        color = palette.get(lab, "white")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = f"{lab} {sc:.2f}"
        tw, th = _measure(draw, caption, font)
        y0 = max(0, y1 - th - 2)
        draw.rectangle([x1, y0, x1 + tw + 4, y0 + th + 2], fill=color)
        draw.text((x1 + 2, y0 + 1), caption, fill="black", font=font)
    return vis

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def is_under(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except AttributeError:
        c, p = child.resolve(), parent.resolve()
        return str(c).startswith(str(p) + os.sep) or c == p

def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "model_id": MODEL_ID, "entries": {}}

def file_signature(p: Path) -> dict:
    st = p.stat()
    sig = {"size": st.st_size, "mtime": int(st.st_mtime)}
    if USE_HASH:
        h = hashlib.blake2s(digest_size=16)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sig["hash"] = h.hexdigest()
    return sig

def should_skip(relpath: str, src_path: Path, manifest: dict) -> bool:
    if FORCE_REPROCESS: return False
    e = manifest["entries"].get(relpath)
    if not e: return False
    sig = file_signature(src_path)
    return e.get("status") in {"annotated", "corrected"} and all(e.get(k) == sig.get(k) for k in sig.keys())

def mark_in_progress(manifest: dict, relpath: str, src_path: Path):
    lbl_rel = str((OUT_LABELS / Path(relpath)).with_suffix(".txt").relative_to(OUT_ROOT))
    vis_rel = str((OUT_VIS / Path(relpath).parent / (Path(relpath).stem + "_viz.jpg")).relative_to(OUT_ROOT))
    manifest["entries"][relpath] = {
        **file_signature(src_path),
        "status": "in_progress",
        "labels": lbl_rel,
        "viz": vis_rel,
        "model": MODEL_ID,
        "score_threshold": SCORE_THRESHOLD,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

def mark_done(manifest: dict, relpath: str, src_path: Path, detections: int, classes: List[str]):
    e = manifest["entries"].setdefault(relpath, {})
    e.update({
        **file_signature(src_path),
        "status": "annotated",
        "detections": int(detections),
        "classes": sorted(set(classes)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

def prune_missing(manifest: dict, src_root: Path):
    keep = {}
    for rel, e in manifest["entries"].items():
        if (src_root / rel).exists():
            keep[rel] = e
    manifest["entries"] = keep

def safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return dst
        except FileNotFoundError:
            pass
        # keep existing; comment next line and uncomment copy2 if you prefer overwrite
        return dst
        # shutil.copy2(src, dst); return dst
    shutil.copy2(src, dst)
    return dst

# -------------------
# MODEL
# -------------------
print("Loading model:", MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
print(f"Processor: {processor.image_processor.__class__.__name__}; size cfg = {getattr(processor.image_processor, 'size', None)}")

# -------------------
# BUILD FILE LIST (exclude OUT_ROOT)
# -------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
all_images_abs = sorted([p for p in SRC_DIR.rglob("*")
                         if p.suffix.lower() in IMAGE_EXTS and not is_under(p, OUT_ROOT)])

yolo_name_to_id = {name: i for i, name in enumerate(YOLO_CLASSES)}
autocast_ctx = torch.autocast("cuda") if DEVICE == "cuda" else nullcontext()

# -------------------
# MANIFEST
# -------------------
manifest = load_manifest()
prune_missing(manifest, SRC_DIR)  # drop entries whose source file disappeared
atomic_write_json(manifest, MANIFEST_PATH)

# -------------------
# MAIN LOOP
# -------------------
num_ok = num_fail = 0

for src_path in all_images_abs:
    rel = str(src_path.relative_to(SRC_DIR))      # manifest key; also preserves folders in outputs

    if should_skip(rel, src_path, manifest):
        continue

    # Ensure nested folders exist in outputs
    (OUT_IMAGES / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
    (OUT_LABELS / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
    (OUT_VIS    / Path(rel)).parent.mkdir(parents=True, exist_ok=True)

    mark_in_progress(manifest, rel, src_path)
    atomic_write_json(manifest, MANIFEST_PATH)

    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"[skip] {src_path} ({e})")
        num_fail += 1
        continue

    W0, H0 = img.size

    # Collect across label chunks
    all_boxes, all_scores, all_names = [], [], []

    with torch.inference_mode(), autocast_ctx:
        for label_batch in chunked(ZS_LABELS, LABEL_CHUNK):
            inputs = processor(images=img, text=[label_batch], return_tensors="pt")
            inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Map OUTPUT boxes directly to ORIGINAL (H0, W0) coordinates
            target_sizes = torch.tensor([(H0, W0)], device=DEVICE)
            res = processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                          threshold=SCORE_THRESHOLD)[0]

            boxes = res["boxes"]                     # tensor [N,4]
            scores = res["scores"].tolist()
            labs = res["labels"]
            if isinstance(labs, torch.Tensor):       # indices per current chunk
                labs = [label_batch[int(i)] for i in labs.tolist()]

            if len(boxes) > 0:
                all_boxes.append(boxes.to("cpu"))
                all_scores.extend(scores)
                all_names.extend(labs)

    # Save/copy image to OUT_IMAGES (path-preserving)
    dst_img = OUT_IMAGES / Path(rel)
    safe_copy(src_path, dst_img)

    # If nothing detected, still create an empty label file so CVAT/YOLO match filenames
    lbl_path = OUT_LABELS / Path(rel).with_suffix(".txt")
    if len(all_boxes) == 0:
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        open(lbl_path, "w").close()
        mark_done(manifest, rel, src_path, detections=0, classes=[])
        atomic_write_json(manifest, MANIFEST_PATH)
        num_ok += 1
        continue

    # Concatenate chunked outputs
    boxes_orig = torch.cat(all_boxes, dim=0).float()  # already in original coords
    boxes_orig = clamp_xyxy(boxes_orig, W0, H0)
    boxes_xyxy = boxes_orig.round().int().tolist()
    scores = all_scores
    labels_raw = all_names

    # Map to canonical training names via ALIASES and filter to YOLO_CLASSES
    kept_xyxy, kept_scores, kept_names = [], [], []
    for (x1,y1,x2,y2), lab, sc in zip(boxes_xyxy, labels_raw, scores):
        cname = ALIASES.get(lab, lab)
        if cname not in YOLO_CLASSES:
            continue
        kept_xyxy.append((x1,y1,x2,y2))
        kept_scores.append(sc)
        kept_names.append(cname)

    # Write YOLO .txt (path-preserving)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w") as f:
        for (x1,y1,x2,y2), cname in zip(kept_xyxy, kept_names):
            cls_id = yolo_name_to_id[cname]
            line = to_yolo_line((x1,y1,x2,y2), cls_id, W0, H0)
            if line: f.write(line + "\n")

    # Visualization (path-preserving)
    vis_path = OUT_VIS / Path(rel).parent / (Path(rel).stem + "_viz.jpg")
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    vis = draw_boxes(img, kept_xyxy, kept_names, kept_scores, min_score=SCORE_THRESHOLD)
    vis.save(vis_path)

    # Update manifest
    mark_done(manifest, rel, src_path, detections=len(kept_xyxy), classes=kept_names)
    atomic_write_json(manifest, MANIFEST_PATH)

    num_ok += 1

print(f"Done. OK: {num_ok}, failed: {num_fail}")

# -------------------
# Optional: quick train/val split (CVAT can also import a flat YOLO folder)
# (When using path-preserving outputs, splitting moves subtrees accordingly.)
if VAL_FRACTION > 0:
    (OUT_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted([p for p in OUT_IMAGES.rglob("*") if p.suffix.lower() in exts])
    random.seed(123)
    val_n = int(len(imgs) * VAL_FRACTION)
    val_set = set(random.sample(imgs, val_n))

    for img_p in imgs:
        rel_img = img_p.relative_to(OUT_IMAGES)
        lbl_p = OUT_LABELS / rel_img.with_suffix(".txt")
        split = "val" if img_p in val_set else "train"

        # Ensure subdirs exist
        (OUT_ROOT / "images" / split / rel_img.parent).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split / rel_img.parent).mkdir(parents=True, exist_ok=True)

        shutil.move(str(img_p), str(OUT_ROOT / "images" / split / rel_img))
        shutil.move(str(lbl_p), str(OUT_ROOT / "labels" / split / rel_img.with_suffix(".txt")))

# -------------------
# Write a minimal data.yaml for YOLO
data_yaml = OUT_ROOT / "data.yaml"
with open(data_yaml, "w") as f:
    if VAL_FRACTION > 0:
        f.write(
            f"path: {OUT_ROOT}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )
    else:
        f.write(
            f"path: {OUT_ROOT}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )

print(f"Wrote data.yaml -> {data_yaml}")
print("CVAT tip: Task → Import → Format: 'YOLO 1.1' → select images/ + labels/")
print(f"Manifest: {MANIFEST_PATH}  (status: annotated/in_progress/corrected)")
