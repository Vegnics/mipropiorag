from paddleocr import PaddleOCR
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from lang_utils.raw_text_ops2 import clean_slide_text,clean_page_text
device = "cuda" if torch.cuda.is_available() else "cpu"
"""
proc0 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model0 = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device)

image_np = cv2.imread(iname)
image = Image.fromarray(image_np)
inputs = proc0(images=image,return_tensors="pt").to(device)
#inputs["input_ids"]=None
with torch.no_grad():
    out = model0.generate(**inputs, max_new_tokens=40)
caption = proc0.decode(out[0], skip_special_tokens=True)
## Block diagram
#if "block" in caption and "diagram" in caption or True:
"""


def compute_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter + 1e-6
    return inter / union


# -------------------------------
# Deduplication
# -------------------------------
def deduplicate(items, iou_thresh=0.5):
    kept = []
    for item in items:
        duplicate = False
        for k in kept:
            same_text = item["text"].strip().lower() == k["text"].strip().lower()
            iou = compute_iou(item, k)
            if same_text and iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


# -------------------------------
# Patch generation
# -------------------------------
def generate_patch_coords(H, W, ph, pw, sh, sw):
    ys = list(range(0, max(H - ph + 1, 1), sh))
    xs = list(range(0, max(W - pw + 1, 1), sw))
    if ys[-1] != max(H - ph, 0):
        ys.append(max(H - ph, 0))
    if xs[-1] != max(W - pw, 0):
        xs.append(max(W - pw, 0))
    return ys, xs



class OCR_Reader:
    def __init__(self,y_thresh=100):
        self.y_thresh = y_thresh
        self.ocr_model = PaddleOCR(use_angle_cls=False, lang="en")
    def _raw_paddle_ocr(self,img):
        results = self.ocr_model.predict(img)
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

    # -------------------------------
    # Main tiled OCR function
    # -------------------------------
    def tiled_ocr_collect(self,image,
                        patch_h=512, patch_w=512,
                        stride_h=384, stride_w=384):

        H, W = image.shape[:2]
        all_items = []
        patch_h = int(H*0.4)
        patch_w = int(W*0.4)
        stride_h = int(patch_h*0.8)
        stride_w = int(patch_w*0.8)
        ys, xs = generate_patch_coords(H, W, patch_h, patch_w, stride_h, stride_w)
        for y in ys:
            for x in xs:
                patch = image[y:y + patch_h, x:x + patch_w]

                patch_items = self._raw_paddle_ocr(patch)

                for item in patch_items:
                    global_item = {
                        "text": item["text"],
                        "bbox": [[p[0] + x, p[1] + y] for p in item["bbox"]],
                        "x1": item["x1"] + x,
                        "y1": item["y1"] + y,
                        "x2": item["x2"] + x,
                        "y2": item["y2"] + y,
                        "cx": item["cx"] + x,
                        "cy": item["cy"] + y,
                    }

                    all_items.append(global_item)

        # deduplicate overlapping detections
        all_items = deduplicate(all_items)

        return all_items
    def group_into_rows(self,items):
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

            if abs(item["cy"] - avg_y) <= self.y_thresh:
                current_row.append(item)
            else:
                current_row = sorted(current_row, key=lambda x: x["x1"])
                rows.append(current_row)
                current_row = [item]

        if current_row:
            current_row = sorted(current_row, key=lambda x: x["x1"])
            rows.append(current_row)

        return rows

    def rows_to_text(self,rows):
        lines = []
        for row in rows:
            line_text = " ".join(item["text"] for item in row)
            lines.append(line_text)
        return "\n".join(lines)

    def image2text_ocr(self,image_np):
        data = self._raw_paddle_ocr(image_np) #self.raw_paddle_ocr(image_np)
        rows = self.group_into_rows(data)
        text = self.rows_to_text(rows)
        return text


class OCR_Reader2:
    def __init__(self,y_thresh=100):
        self.y_thresh = y_thresh
        self.ocr_model = PaddleOCR(use_angle_cls=False, lang="en")
    def _raw_paddle_ocr(self,img):
        results = self.ocr_model.predict(img)
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

    # -------------------------------
    # Main tiled OCR function
    # -------------------------------
    def tiled_ocr_collect(self,image,
                        patch_h=512, patch_w=512,
                        stride_h=384, stride_w=384):

        H, W = image.shape[:2]
        all_items = []
        patch_h = int(H*0.4)
        patch_w = int(W*0.4)
        stride_h = int(patch_h*0.8)
        stride_w = int(patch_w*0.8)
        ys, xs = generate_patch_coords(H, W, patch_h, patch_w, stride_h, stride_w)
        for y in ys:
            for x in xs:
                patch = image[y:y + patch_h, x:x + patch_w]

                patch_items = self._raw_paddle_ocr(patch)

                for item in patch_items:
                    global_item = {
                        "text": item["text"],
                        "bbox": [[p[0] + x, p[1] + y] for p in item["bbox"]],
                        "x1": item["x1"] + x,
                        "y1": item["y1"] + y,
                        "x2": item["x2"] + x,
                        "y2": item["y2"] + y,
                        "cx": item["cx"] + x,
                        "cy": item["cy"] + y,
                    }

                    all_items.append(global_item)

        # deduplicate overlapping detections
        all_items = deduplicate(all_items)

        return all_items
    def group_into_rows(self,items):
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

            if abs(item["cy"] - avg_y) <= self.y_thresh:
                current_row.append(item)
            else:
                current_row = sorted(current_row, key=lambda x: x["x1"])
                rows.append(current_row)
                current_row = [item]

        if current_row:
            current_row = sorted(current_row, key=lambda x: x["x1"])
            rows.append(current_row)

        return rows

    def image2rows_ocr(self, image_np):
        data = self._raw_paddle_ocr(image_np)
        rows = self.group_into_rows(data)
        return rows
    
    def rows_to_text(self,rows):
        lines = []
        for row in rows:
            line_text = " ".join(item["text"] for item in row)
            lines.append(line_text)
        return "\n".join(lines)
    
    def rows_to_lines(self, rows):
        lines = []
        for row in rows:
            line_text = " ".join(item["text"] for item in row).strip()
            if line_text:
                line_text = clean_page_text(line_text)
                lines.append(line_text)
        return lines
    

    def image2text_ocr(self,image_np):
        data = self._raw_paddle_ocr(image_np) #self.raw_paddle_ocr(image_np)
        rows = self.group_into_rows(data)
        lines = self.rows_to_lines(rows)
        return lines

# Example usage

#print(caption,"\n",text)