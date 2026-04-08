from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import Qwen2VLForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"
proc1 = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model1 = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

image = Image.open("examplevqa.png")
question = f"AI Lecture. The table explains .."
inputs = proc1(images=image,text=question, return_tensors="pt").to(device)
with torch.no_grad():
    out = model1.generate(**inputs, max_new_tokens=40)
caption = "Block Diagram: "+ proc1.decode(out[0], skip_special_tokens=True)
print(caption)