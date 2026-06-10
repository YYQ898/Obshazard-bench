import os
import json
import argparse
from collections import Counter
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, required=True, help="Path to the JSON dataset file")
parser.add_argument("--image-base-path", type=str, required=True, help="Base path for images")
args = parser.parse_args()

dataset_path = args.dataset_path
image_base_path = args.image_base_path
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

image_num_list = []
task_list = []
subtask_list = []
image_size_list = []
for item in tqdm(dataset):
    images_str = item.get("Image", "")
    task = item.get("Task", "")
    subtask = item.get("Subtask", "")
    paths = [p.strip() for p in images_str.split(',') if p.strip()]
    for path in paths:
        full_path = os.path.join(image_base_path, path)
        with Image.open(full_path) as img:
            width, height = img.size
            image_size_list.append((width, height))
        
    image_num = len(paths)  
    image_num_list.append(image_num)
    task_list.append(task)
    subtask_list.append(subtask)

# 统计 task 分布
task_dist = Counter(task_list)
print("------Task 分布（任务名: 数量）:------")
for t, cnt in sorted(task_dist.items()):
    print(f"{t}: {cnt} ({cnt/sum(task_dist.values())*100:.2f}%)")

subtask_dist = Counter(subtask_list)
print("------Subtask 分布（子任务名: 数量）:------")
for s, cnt in sorted(subtask_dist.items()):
    print(f"{s}: {cnt} ({cnt/sum(subtask_dist.values())*100:.2f}%)")

image_num_dist = Counter(image_num_list)
total_num = sum(image_num_dist.values())
print(f"总样本数: {total_num}")
print("------图片数量分布（数量: 样本数）:------")
for num, cnt in sorted(image_num_dist.items()):
    print(f"{num}: {cnt} ({cnt/total_num*100:.2f}%)")

# 统计 image size 分布
image_size_dist = Counter(image_size_list)
print("------Image size 分布（宽度x高度: 数量）:------")
for size, cnt in sorted(image_size_dist.items()):
    print(f"{size}: {cnt} ({cnt/sum(image_size_dist.values())*100:.2f}%)")