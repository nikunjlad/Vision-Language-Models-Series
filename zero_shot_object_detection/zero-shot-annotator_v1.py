
# import necessary libraries
import os
# from ultralytics import YOLO
from transformers import pipeline
from PIL import Image, ImageDraw
from pathlib import Path

# define zero shot detector pipeline
detector = pipeline('zero-shot-object-detection', model='google/owlv2-base-patch16-ensemble')

# classes for detection
classes = ['man', 'soldier', 'woman', 'vehicle', 'weapon', 'street_sign', 'animal']
class2id = {c:i for i,c in enumerate(classes)}
colors = ['red', 'red', 'red', 'blue', 'green', 'yellow', 'purple']

# data folders
src_folder = Path('/home/nikunj/research/Vision-Language-Models-Series/data/zero-shot-test-images')
out_folder = Path('/home/nikunj/research/Vision-Language-Models-Series/data/zero-shot-test-images/zero_shot_annotated')
os.makedirs(out_folder, exist_ok=True)


for fname in sorted(src_folder.rglob("*.jpeg")):
    if not str(fname).lower().endswith('.jpeg'):
        continue

    img = Image.open(str(fname))
    draw = ImageDraw.Draw(img)

    # run detector
    detections = detector(img, candidate_labels=classes, threshold=0.3)

    w, h = img.size
    lines = []
    for det in detections:
        cid = class2id[det['label']]
        output = det['box']
        xmin, ymin, xmax, ymax = int(output['xmin']), int(output['ymin']), int(output['xmax']), int(output['ymax'])

        draw.rectangle([xmin, ymin, xmax, ymax], outline=colors[cid], width=3)

    img.show()
