import os
import json


# json_file1 = "eval_result_20260225_152811.json"
# data_name1 = set()
# with open(json_file1, "r") as f:
#     data = json.load(f)
#     for item in data["details"]:
#         data_name1.add(item["id"])

# json_file2 = "eval_result_20260225_161104.json"
# data_name2 = set()
# with open(json_file2, "r") as f:
#     data = json.load(f)
#     for item in data["details"]:
#         data_name2.add(item["id"])
        
# # 比较两个集合的差集
# diff1 = data_name1 - data_name2
# diff2 = data_name2 - data_name1
# print("只在json_file1中的id:", diff1)
# print("只在json_file2中的id:", diff2)


# json_file = "output/eval_result_earthquake0201_Image_Only.json_20260226_205353.json"
json_file = "output/eval_result_earthquake0201_gemini-3-pro-preview-thinking_20260227_132500.json"
with open(json_file, "r") as f:
    data = json.load(f)
    max_len = 0
    max_reasoning_content_len = 0
    len0_num = 0
    for item in data["details"]:
        max_len = max(max_len, len(item['raw_response']))
        if 'raw_reasoning_content' in item:
            max_reasoning_content_len = max(max_reasoning_content_len, len(item['raw_reasoning_content']))
        if len(item['raw_response']) == 0:
            len0_num += 1
    print(max_len)
    print(len0_num)
    print(len(data["details"]))
    print(max_reasoning_content_len)
   