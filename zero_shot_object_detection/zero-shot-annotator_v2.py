"""
Incremental zero-shot pre-annotation -> YOLO (CVAT-friendly) with versioned snapshots.
Directory layout matches the plan with RAW on exFAT and workspace on ext4.

Subcommands:
  - annotate      : pre-annotate NEW/CHANGED images, write YOLO labels, build per-run CVAT zip
  - snapshot      : create frozen dataset (yolo_vN) with symlinked images and copied labels

Example:
  python pipeline.py annotate
  python pipeline.py snapshot --name yolo_v1 --val-ratio 0.1 --corrected-only
"""

import os, sys, json, time, random, shutil, hashlib, tempfile, zipfile, argparse, datetime
from pathlib import Path
from typing import List, Tuple
from contextlib import nullcontext

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Owlv2ForObjectDetection

# -------------------
# PATHS (edit if needed)
# -------------------
RAW_ROOT = Path("/mnt/data/vision-project-raw")                 # EXFAT (read-only to pipeline)
PROJECT_ROOT = Path("~/vision-project").expanduser()            # EXT4
OUT_ROOT = PROJECT_ROOT / "autolabel"                           # pre-annotations
MERGED_ROOT = PROJECT_ROOT / "merged"                           # training-ready live view
SNAPSHOTS_DIR = MERGED_ROOT / "datasets"                        # frozen datasets

OUT_IMAGES = OUT_ROOT / "images"                                # per-file symlinks -> RAW_ROOT/**
OUT_LABELS = OUT_ROOT / "labels"                                # YOLO .txt (real files)
OUT_VIZ    = OUT_ROOT / "viz"
STAGING_DIR = OUT_ROOT / "staging"                              # per-run staging (zip only is okay)
EXPORTS_DIR = OUT_ROOT / "exports_cvat"                         # downloaded CVAT exports

MANIFEST_PATH = OUT_ROOT / "manifest.json"

# Ensure base dirs
for p in (OUT_IMAGES, OUT_LABELS, OUT_VIZ, STAGING_DIR, EXPORTS_DIR, SNAPSHOTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Ensure merged/images -> autolabel/images (dir symlink)
MERGED_IMAGES = MERGED_ROOT / "images"
MERGED_LABELS = MERGED_ROOT / "labels"
MERGED_ROOT.mkdir(parents=True, exist_ok=True)
if not MERGED_IMAGES.exists():
    try:
        MERGED_IMAGES.symlink_to(Path("..") / "autolabel" / "images")  # relative symlink
    except OSError:
        # fallback: absolute symlink
        MERGED_IMAGES.symlink_to(OUT_IMAGES)

MERGED_LABELS.mkdir(parents=True, exist_ok=True)

# -------------------
# MODEL / ZS CONFIG
# -------------------
MODEL_ID = "google/owlv2-base-patch16-ensemble"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical YOLO classes you want to train on
YOLO_CLASSES = ["person", "vehicle", "street_sign", "weapon"]

# Zero-shot prompts (keep concise for bootstrapping)
ZS_LABELS = [
    "man", "woman", "soldier",
    "car", 'bicycle', 'motorcycle', 'train', 'bus', 'truck',
    "weapon", "knife", "shotgun", "rifle", "pistol", "gun",
    "street_sign",
]

# Map synonyms/phrases → canonical training classes (skip those that don't map)
ALIASES = {
    "man": "person",
    "woman": "person",
    "soldier": "person",
    "car": "vehicle",
    "bicycle": "vehicle",
    "motorcycle": "vehicle",
    "train": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "street_sign": "street_sign",
    "weapon": "weapon",
    "gun": "weapon",
    "pistol": "weapon",
    "rifle": "weapon",
    "shotgun": "weapon",
    "knife": "weapon",
}

SCORE_THRESHOLD = 0.30
LABEL_CHUNK = 24
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Storage policy
SAVE_VIZ = False          # previews can be big; toggle on if you need them
VIZ_TTL_DAYS = 7
KEEP_IMPORT_ZIPS = 2      # keep last N staging zips; set 0 to delete right away

# -------------------
# UTILITIES
# -------------------
def is_under(child: Path, parent: Path) -> bool:
    child, parent = Path(child).resolve(), Path(parent).resolve()
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

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
    palette = {"person":"red","vehicle":"blue","street_sign":"yellow","weapon":"green"}
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

def file_signature(p: Path, use_hash=False) -> dict:
    st = p.stat()
    sig = {"size": st.st_size, "mtime": int(st.st_mtime)}
    if use_hash:
        h = hashlib.blake2s(digest_size=16)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sig["hash"] = h.hexdigest()
    return sig

def should_skip(relpath: str, src_path: Path, manifest: dict, force=False, use_hash=False) -> bool:
    if force: return False
    e = manifest["entries"].get(relpath)
    if not e: return False
    sig = file_signature(src_path, use_hash=use_hash)
    return e.get("status") in {"annotated", "corrected"} and all(e.get(k) == sig.get(k) for k in sig.keys())

def mark_in_progress(manifest: dict, relpath: str, src_path: Path):
    lbl_rel = str((OUT_LABELS / Path(relpath)).with_suffix(".txt").relative_to(OUT_ROOT))
    viz_rel = str((OUT_VIZ / Path(relpath).parent / (Path(relpath).stem + "_viz.jpg")).relative_to(OUT_ROOT))
    manifest["entries"][relpath] = {
        **file_signature(src_path),
        "status": "in_progress",
        "labels": lbl_rel,
        "viz": viz_rel,
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

def symlink_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        dst.symlink_to(src)   # symlink lives on ext4, points to exFAT raw
    except OSError:
        shutil.copy2(src, dst)  # fallback
    return dst

def gc_old_viz(ttl_days=VIZ_TTL_DAYS):
    if not SAVE_VIZ: return
    cutoff = time.time() - ttl_days*86400
    for p in OUT_VIZ.rglob("*.jpg"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            pass

def build_cvat_batch_zip(run_id: str, relpaths: List[str]) -> Path:
    """Zip only the NEW images+labels (no staging copies)."""
    zpath = STAGING_DIR / f"cvat_batch_{run_id}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # labels from OUT_LABELS
        for rel in relpaths:
            lbl = OUT_LABELS / Path(rel).with_suffix(".txt")
            if lbl.exists():
                arc = Path("labels") / Path(rel).with_suffix(".txt")
                z.write(lbl, arcname=str(arc))
        # images from RAW_ROOT
        for rel in relpaths:
            img = RAW_ROOT / rel
            if img.exists():
                arc = Path("images") / rel
                z.write(img, arcname=str(arc))
        # Optional: data.yaml
        z.writestr("data.yaml",
                   f"path: .\ntrain: images\nval: images\nnc: {len(YOLO_CLASSES)}\n"
                   f"names: {YOLO_CLASSES}\n")
    return zpath

# -------------------
# MODEL
# -------------------
def load_model():
    print("Loading model:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return processor, model

# -------------------
# COMMAND: annotate
# -------------------
def cmd_annotate(args):
    # Build source list (RAW_ROOT only)
    all_images = sorted([p for p in RAW_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not all_images:
        print(f"No images found under {RAW_ROOT}")
        return

    manifest = load_manifest()
    prune_missing(manifest, RAW_ROOT)
    atomic_write_json(manifest, MANIFEST_PATH)

    processor, model = load_model()
    autocast_ctx = torch.autocast("cuda") if DEVICE == "cuda" else nullcontext()
    yolo_name_to_id = {name: i for i, name in enumerate(YOLO_CLASSES)}

    run_id = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    new_this_run: List[str] = []

    num_ok = num_fail = 0
    for src_path in all_images:
        rel = str(src_path.relative_to(RAW_ROOT))
        if should_skip(rel, src_path, manifest, force=args.force, use_hash=args.hash):
            continue

        # ensure mirror dirs
        (OUT_IMAGES / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
        (OUT_LABELS / Path(rel)).parent.mkdir(parents=True, exist_ok=True)
        (OUT_VIZ    / Path(rel)).parent.mkdir(parents=True, exist_ok=True)

        mark_in_progress(manifest, rel, src_path)
        atomic_write_json(manifest, MANIFEST_PATH)

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {src_path} ({e})")
            num_fail += 1
            continue

        W0, H0 = img.size
        all_boxes, all_scores, all_names = [], [], []

        with torch.inference_mode(), autocast_ctx:
            for label_batch in chunked(ZS_LABELS, LABEL_CHUNK):
                inputs = processor(images=img, text=[label_batch], return_tensors="pt")
                inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                outputs = model(**inputs)
                target_sizes = torch.tensor([(H0, W0)], device=DEVICE)
                res = processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                              threshold=SCORE_THRESHOLD)[0]
                boxes = res["boxes"]
                scores = res["scores"].tolist()
                labs = res["labels"]
                if isinstance(labs, torch.Tensor):
                    labs = [label_batch[int(i)] for i in labs.tolist()]
                if len(boxes) > 0:
                    all_boxes.append(boxes.to("cpu"))
                    all_scores.extend(scores)
                    all_names.extend(labs)

        # symlink image into OUT_IMAGES (path-preserving)
        symlink_file(src_path, OUT_IMAGES / Path(rel))

        # labels path
        lbl_path = OUT_LABELS / Path(rel).with_suffix(".txt")

        if len(all_boxes) == 0:
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            open(lbl_path, "w").close()
            mark_done(manifest, rel, src_path, detections=0, classes=[])
            atomic_write_json(manifest, MANIFEST_PATH)
            new_this_run.append(rel)
            num_ok += 1
            continue

        boxes_orig = torch.cat(all_boxes, dim=0).float()
        boxes_orig = clamp_xyxy(boxes_orig, W0, H0)
        boxes_xyxy = boxes_orig.round().int().tolist()
        scores = all_scores
        labels_raw = all_names

        kept_xyxy, kept_scores, kept_names = [], [], []
        for (x1,y1,x2,y2), lab, sc in zip(boxes_xyxy, labels_raw, scores):
            cname = ALIASES.get(lab, lab)
            if cname not in YOLO_CLASSES:
                continue
            kept_xyxy.append((x1,y1,x2,y2))
            kept_scores.append(sc)
            kept_names.append(cname)

        # write YOLO
        with open(lbl_path, "w") as f:
            for (x1,y1,x2,y2), cname in zip(kept_xyxy, kept_names):
                cls_id = yolo_name_to_id[cname]
                line = to_yolo_line((x1,y1,x2,y2), cls_id, W0, H0)
                if line: f.write(line + "\n")

        # optional viz
        if SAVE_VIZ:
            vis_path = OUT_VIZ / Path(rel).parent / (Path(rel).stem + "_viz.jpg")
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            vis = draw_boxes(img, kept_xyxy, kept_names, kept_scores, min_score=SCORE_THRESHOLD)
            vis.save(vis_path)

        mark_done(manifest, rel, src_path, detections=len(kept_xyxy), classes=kept_names)
        atomic_write_json(manifest, MANIFEST_PATH)
        new_this_run.append(rel)
        num_ok += 1

    # Build per-run CVAT zip with only NEW files
    if new_this_run:
        run_zip = build_cvat_batch_zip(run_id, new_this_run)
        print(f"\nStaging zip ready for CVAT:\n  {run_zip}")
        # GC old import zips
        if KEEP_IMPORT_ZIPS >= 0:
            zips = sorted(STAGING_DIR.glob("cvat_batch_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in zips[KEEP_IMPORT_ZIPS:]:
                try: old.unlink()
                except: pass
    else:
        print("\nNo new/changed files this run; no staging zip created.")

    # GC old viz (optional)
    gc_old_viz()

    # Write data.yaml for autolabel/
    with open(OUT_ROOT / "data.yaml", "w") as f:
        f.write(
            f"path: {OUT_ROOT}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )

    print(f"\nDone. OK: {num_ok}, failed: {num_fail}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("CVAT: Create a NEW task, Upload Dataset → 'YOLO 1.1' → select the staging zip above.")

# -------------------
# COMMAND: snapshot
# -------------------
def cmd_snapshot(args):
    """
    Create a frozen dataset under merged/datasets/<name> with:
      - images/train|val symlinked to merged/images/**
      - labels/train|val copied from merged/labels/** (corrected overwrite policy assumed upstream)
    Split can be either:
      - fixed filelists provided via --train-list/--val-list, or
      - a random split via --val-ratio (default 0.1)
    """
    snap_name = args.name.strip()
    SNAP = SNAPSHOTS_DIR / snap_name
    if SNAP.exists():
        print(f"[snapshot] {SNAP} already exists; refusing to overwrite.")
        sys.exit(1)

    # Read manifest to decide which files to include
    manifest = load_manifest().get("entries", {})
    candidates = []
    for rel, meta in manifest.items():
        st = meta.get("status")
        if args.corrected_only:
            if st == "corrected":
                # include only if both image and label exist in merged view
                img = MERGED_IMAGES / rel
                lbl = MERGED_LABELS / Path(rel).with_suffix(".txt")
                if img.exists() and lbl.exists():
                    candidates.append(rel)
        else:
            if st in {"annotated", "corrected"}:
                img = MERGED_IMAGES / rel
                lbl = MERGED_LABELS / Path(rel).with_suffix(".txt")
                if img.exists() and lbl.exists():
                    candidates.append(rel)

    if not candidates:
        print("[snapshot] No eligible files found. Did you merge labels into merged/labels/?")
        sys.exit(1)

    # Build lists
    if args.train_list and args.val_list:
        with open(args.train_list, "r") as f:
            train_list = [ln.strip() for ln in f if ln.strip()]
        with open(args.val_list, "r") as f:
            val_list = [ln.strip() for ln in f if ln.strip()]
    else:
        random.seed(42)
        random.shuffle(candidates)
        val_n = int(len(candidates) * args.val_ratio)
        val_list = candidates[:val_n]
        train_list = candidates[val_n:]

    # Lay out snapshot dirs
    (SNAP / "images" / "train").mkdir(parents=True, exist_ok=True)
    (SNAP / "images" / "val").mkdir(parents=True, exist_ok=True)
    (SNAP / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (SNAP / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Symlink images; copy labels
    def link_images(rel_list, split):
        for rel in rel_list:
            src = MERGED_IMAGES / rel               # symlink in merged/images -> raw
            dst = SNAP / "images" / split / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst.symlink_to(Path("../../../../") / "images" / rel)  # relative path within snapshot
            except OSError:
                # fallback: absolute link
                try:
                    dst.symlink_to(src)
                except OSError:
                    # last resort: copy (not ideal, but safe)
                    shutil.copy2(src, dst)

    def copy_labels(rel_list, split):
        for rel in rel_list:
            src = MERGED_LABELS / Path(rel).with_suffix(".txt")
            dst = SNAP / "labels" / split / Path(rel).with_suffix(".txt")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    link_images(train_list, "train")
    link_images(val_list, "val")
    copy_labels(train_list, "train")
    copy_labels(val_list, "val")

    # data.yaml for the snapshot
    with open(SNAP / "data.yaml", "w") as f:
        f.write(
            f"path: {SNAP}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(YOLO_CLASSES)}\n"
            f"names: {YOLO_CLASSES}\n"
        )

    print(f"[snapshot] Created: {SNAP}")
    print(f"  images/train -> symlinks into {MERGED_IMAGES}")
    print(f"  labels/train -> copied from {MERGED_LABELS}")
    print("Train YOLO on:", SNAP / "data.yaml")

# -------------------
# MAIN
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Incremental pre-annotation + snapshots")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Pre-annotate NEW images and build a per-run CVAT zip")
    p_ann.add_argument("--force", action="store_true", help="Ignore manifest and reprocess everything")
    p_ann.add_argument("--hash", action="store_true", help="Use content hash (slower) for change detection")
    p_ann.set_defaults(func=cmd_annotate)

    p_snap = sub.add_parser("snapshot", help="Create a frozen dataset (yolo_vN)")
    p_snap.add_argument("--name", required=True, help="Snapshot folder name, e.g., yolo_v1")
    p_snap.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction if no filelists provided")
    p_snap.add_argument("--train-list", type=str, default=None, help="Optional filelist of rel image paths for train")
    p_snap.add_argument("--val-list", type=str, default=None, help="Optional filelist of rel image paths for val")
    p_snap.add_argument("--corrected-only", action="store_true", help="Use only CVAT-corrected items")
    p_snap.set_defaults(func=cmd_snapshot)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
