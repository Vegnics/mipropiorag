import json

text_info_file = "slides_dict_3.json"
slide_info_file = "slides_dict_descriptions.json"


with open(text_info_file,'r') as f:
    tinfo = json.load(f)

with open(slide_info_file,'r') as f:
    sinfo = json.load(f)

outdict = {}
for k,(tv,sv) in enumerate(zip(tinfo.values(),sinfo.values())):
    print(tv,sv)
    outdict[k] = {}
    outdict[k]["slide_desc"] = sv["slide_desc"]
    outdict[k]["title"] = tv["title"]
    outdict[k]["body"] = tv["body"] 
    outdict[k]["imgtext"] = tv["imgtext"]
    outdict[k]["reference"] = tv["reference"]

with open("full_slides_dict.json",'w') as f:
    json.dump(outdict,f)