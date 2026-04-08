import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import cv2

model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
img_np = cv2.imread("examplevqa2.png")
processor = AutoProcessor.from_pretrained(model_name)
img = Image.fromarray(img_np.astype(np.uint8))

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What is in this image?"}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(out, skip_special_tokens=True)[0])