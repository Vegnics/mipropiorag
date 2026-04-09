import json
from copy import deepcopy

STRUCT_PATH = "/home/amaranth2/Downloads/slides_dict_new.json"
DESC_PATH = "/home/amaranth2/Downloads/slides_extended_desc.json"
OUT_PATH = "slides_merged.json"

with open(STRUCT_PATH, "r", encoding="utf-8") as f:
    slides_struct = json.load(f)

with open(DESC_PATH, "r", encoding="utf-8") as f:
    slides_desc = json.load(f)

merged = {}

for key, slide in slides_struct.items():
    slide_out = deepcopy(slide)

    desc_entry = slides_desc.get(str(key), {})
    qa_descs = desc_entry.get("qa_descs", {})

    
    slide_out["qa_descs"] = qa_descs
    slide_out["desc_chunks"] = []

    del slide_out["desc_text"]
    del slide_out["title_text"]
    del slide_out["body_text"]
    del slide_out["ocr_text"]
    del slide_out["desc_chunks"]

    merged[key] = slide_out

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Saved merged file to {OUT_PATH}")