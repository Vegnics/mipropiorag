from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import cv2

model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
img_np = cv2.imread("examplevqa2.png")
img = Image.fromarray(img_np.astype("uint8"))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=prompt,
    images=[img],
    return_tensors="pt"
).to(model.device)

out = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(out, skip_special_tokens=True)[0])