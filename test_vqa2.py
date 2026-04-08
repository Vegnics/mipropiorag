from paddleocr import PaddleOCR
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cpu"

proc0 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model0 = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device)

image_np = cv2.imread("examplevqa.png")
image = Image.fromarray(image_np)
inputs = proc0(images=image,return_tensors="pt").to(device)
#inputs["input_ids"]=None
with torch.no_grad():
    out = model0.generate(**inputs, max_new_tokens=40)
caption = proc0.decode(out[0], skip_special_tokens=True)
## Block diagram
#if "block" in caption and "diagram" in caption or True:

def paddle_ocr_extract(image_path):
    ocr = PaddleOCR(use_angle_cls=False, lang="en")
    img = cv2.imread(image_path)
    results = ocr.ocr(img)

    extracted = []

    for line in results:
        texts = line["rec_texts"]
        boxes = line["rec_boxes"]
        for txt, box in zip(texts, boxes):
            box = np.array(box)
            x1 = int(np.min(box[0]))
            y1 = int(np.min(box[1]))
            x2 = int(np.max(box[0]))
            y2 = int(np.max(box[1]))

            extracted.append({
                "text": txt,
                "bbox": box.tolist(),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2
            })

    return extracted
def group_into_rows(items, y_threshold=18):
    # sort top-to-bottom, then left-to-right
    items = sorted(items, key=lambda x: (x["cy"], x["x1"]))

    rows = []
    current_row = []

    for item in items:
        if not current_row:
            current_row.append(item)
            continue

        # compare with average y of current row
        avg_y = sum(x["cy"] for x in current_row) / len(current_row)

        if abs(item["cy"] - avg_y) <= y_threshold:
            current_row.append(item)
        else:
            current_row = sorted(current_row, key=lambda x: x["x1"])
            rows.append(current_row)
            current_row = [item]

    if current_row:
        current_row = sorted(current_row, key=lambda x: x["x1"])
        rows.append(current_row)

    return rows

def rows_to_text(rows):
    lines = []
    for row in rows:
        line_text = " ".join(item["text"] for item in row)
        lines.append(line_text)
    return "\n".join(lines)

# Example usage
data = paddle_ocr_extract("examplevqa2.png")
rows = group_into_rows(data, y_threshold=60)
text = rows_to_text(rows)
print(caption,"\n",text)