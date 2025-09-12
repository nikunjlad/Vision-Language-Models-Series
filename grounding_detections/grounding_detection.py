from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"  # or grounding-dino-base
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval().to("cuda")

image = Image.open("/home/nikunj/research/Vision-Language-Models-Series/data/grounding-test-images/000_1FN8ZQ.jpg").convert("RGB")
draw = ImageDraw.Draw(image)
text_labels = [["a man in a red jacket walking on the pavement"]]  # free-form phrase

with torch.inference_mode():
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.35,           # score threshold
        text_threshold=0.30,      # text alignment threshold
        target_sizes=[image.size[::-1]]
    )[0]

bboxes = results["boxes"].cpu().numpy().astype(int).tolist()
for box in bboxes:
    draw.rectangle(box, outline="red", width=3)
image.show()
print(results["boxes"], results["scores"], results["text_labels"])  # labels echo your phrase
