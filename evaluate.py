import os
import sys
import argparse
import json
import base64
import time
import re
import math
import io
import random
import concurrent.futures
import copy
from typing import Tuple, Optional, Dict, Any, List
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from PIL import Image

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=3600)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=str, required=True, default="")
parser.add_argument("--raw-data-base-path", type=str, required=True, default="")
parser.add_argument("--target-file", type=str, default="Image_Only.json")
parser.add_argument("--output-root", default="./output")
parser.add_argument("--model-name", type=str, default="gpt-4o")
parser.add_argument("--max-workers", type=int, default=64)
parser.add_argument("--total-max-size", type=int, default=int(45 * 1024 * 1024))
parser.add_argument("--image-max-num", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--max-tokens", type=int, default=300)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume-file", type=str, default="")
args = parser.parse_args()

DATASET_ROOT = args.dataset_root
BASE_NAME = os.path.basename(DATASET_ROOT)
RAW_DATA_BASE_PATH = args.raw_data_base_path
TARGET_FILE = args.target_file
OUTPUT_ROOT = args.output_root
MODEL_NAME = args.model_name
MAX_WORKERS = args.max_workers
TOTAL_MAX_SIZE = args.total_max_size  # 45MB
IMAGE_MAX_NUM = args.image_max_num  # max images
temperature = args.temperature
max_tokens = args.max_tokens
resume = args.resume
resume_file = args.resume_file
if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)


def load_metadata(root_dir: str) -> str:
    path = os.path.join(root_dir, "dataset_metadata.json")
    if not os.path.exists(path):
        return ""
    
    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    text = "\n[Domain Knowledge / Metadata]\n"
    
    sat_data = meta.get("Satellite_Metadata", {})
    text += "1. Satellite Sensor Specifications:\n"
    for sat, channels in sat_data.items():
        text += f"   - {sat}:\n"
        sorted_chs = sorted(channels.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
        for ch_id, specs in sorted_chs:
            spec_str = ", ".join([f"{k}={v}" for k, v in specs.items()])
            text += f"     * Band {ch_id}: {spec_str}\n"
            
    st_data = meta.get("Station_Metadata", {})
    text += "\n2. Station Data Definitions:\n"
    for k, v in st_data.get("Descriptions", {}).items():
        text += f"   - {k}: {v}\n"
    
    text += "\n3. Field Abbreviations:\n"
    for k, v in st_data.get("Field_Definitions", {}).items():
        text += f"   - {k} represents: {', '.join(v)}\n"
        
    return text


# def encode_image(image_path: str, max_size: int) -> str:
#     """
#     将图片编码为base64字符串，如果图片超过max_size（字节），则报错或处理。
#     默认最大为4MB。
#     """
#     file_size = os.path.getsize(image_path)
#     if file_size > max_size:
#         # 如果图片超过max_size，则缩小图片，直到小于max_size
#         with Image.open(image_path) as img:
#             ratio = (max_size / file_size) ** 0.5
#             print(f"img_original_size: {img.size}, img_original_bytes_size: {file_size}, ratio: {ratio}")
#             while True:
#                 new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
#                 img = img.resize(new_size)
#                 buffer = io.BytesIO()
#                 img.save(buffer, format="PNG", optimize=True)
#                 img_bytes = buffer.getvalue()
#                 print(f"img_new_size: {img.size}, img_new_bytes_size: {len(img_bytes)}, max_size: {max_size}")
#                 if len(img_bytes) <= max_size:
#                     break
#             buffer.seek(0)
#             return base64.b64encode(buffer.read()).decode('utf-8')
#     else:
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image(image_path: str, max_size: int) -> str:
    """
    将图片编码为base64字符串，如果图片超过max_size（字节），则在不改变图像尺寸的情况下进行压缩（调整JPEG/WEBP等压缩格式质量）。
    """
    # 先获取图片字节数
    file_size = os.path.getsize(image_path)
    # 如果图片字节数超过max_size，则进行压缩
    if file_size > max_size:
        with Image.open(image_path) as img:
            # 优先用JPEG压缩，若不是RGB模式则先转换
            buffer = io.BytesIO()
            quality = 95
            min_quality = 20
            img_for_save = img.convert("RGB") if img.mode != "RGB" else img
            while quality >= min_quality:
                buffer.seek(0)
                # 截断缓冲区 / 文件的内容，只保留指定位置之前的部分，删除该位置之后的所有内容。
                buffer.truncate()
                img_for_save.save(buffer, format="JPEG", quality=quality, optimize=True)
                img_bytes = buffer.getvalue()
                if len(img_bytes) <= max_size:
                    break
                quality -= 5
            else:
                # 如果最小quality还超出max_size，再转为webp继续压缩
                buffer.seek(0)
                buffer.truncate()
                img_for_save.save(buffer, format="WEBP", quality=min_quality, method=6)
                img_bytes = buffer.getvalue()
                if len(img_bytes) > max_size:
                    raise ValueError(f"Image {image_path} cannot be compressed under {max_size} bytes")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def prepare_image_messages(
    image_rel_paths: str, 
    base_path: str, 
    total_max_size: int,
    image_max_num: int
) -> List[Dict[str, Any]]:
    """
    请求输入要求：图片大小总和不可以大于total_max_size，图片数量最多image_max_num
    """
    if not image_rel_paths:
        return []
    
    paths = [p.strip() for p in image_rel_paths.split(',') if p.strip()]
    messages = []
    
    if len(paths) > image_max_num:
        original_num = len(paths)
        paths = paths[:image_max_num]
        print(f"Warning: Image number {original_num} exceeds max_num ({image_max_num}), only use the first {image_max_num} images")
    
    # 计算每张图片的最大字节数, base64编码后的字节数通常是原始的1.33倍，要换算成原始字节数
    max_size_per_image = (total_max_size / 1.33 / 1.33) // len(paths)
    # print(f"Total max size: {total_max_size / 1.33 / 1.33} bytes, Image num: {len(paths)}, Max size per image: {max_size_per_image} bytes")
    total_img_bytes = 0
    for p in paths:
        full_path = os.path.join(base_path, p)
        b64 = encode_image(full_path, max_size_per_image)
        total_img_bytes += len(b64)
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}
        })
    # print(f"Total image bytes after encode image: {total_img_bytes} bytes")
    return messages, paths


def format_station_data(stations: Optional[Dict[str, Any]]) -> str:
    if not stations:
        return "No station data available."
    
    try:
        return json.dumps(stations, ensure_ascii=False, indent=None)
    except:
        return "Error parsing station data."


def normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def try_parse_numeric(s: str) -> Optional[float]:
    # 提取文本中第一个整数 / 浮点数（支持正负号），正则兼顾了「纯整数」「带小数」「带正负号」三种常见数值格式
    m = re.search(r'([-+]?\d+(?:\.\d+)?)', normalize_text(s))
    if m:
        return float(m.group(1))
    return None


def extract_final_answer(model_response: str) -> Dict[str, Any]:
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', model_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            print(f"Error parsing JSON: {json_match.group(1)}")
            pass
            
    try:
        start = model_response.find('{')
        end = model_response.rfind('}')
        if start != -1 and end != -1:
            return json.loads(model_response[start:end+1])
    except:
        print(f"Error parsing JSON: {model_response[start:end+1]}")
        pass
        
    return {"final_answer": model_response.strip(), "answer_type": "text"}


def qa_instance_score(ground_truth: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    gt = str(ground_truth).strip()
    pred = str(extracted.get("final_answer", "")).strip()
    
    if not pred:
        return {"score": 0.0, "reason": "Empty prediction"}

    # boolean
    YES_SET = {"yes", "true", "是"}
    NO_SET = {"no", "false", "否"}
    if gt.lower() in YES_SET or gt.lower() in NO_SET:
        p_norm = pred.lower()
        p_bool = "yes" if any(x in p_norm for x in YES_SET) else ("no" if any(x in p_norm for x in NO_SET) else "unknown")
        g_bool = "yes" if gt.lower() in YES_SET else "no"
        return {"score": 1.0 if p_bool == g_bool else 0.0, "type": "boolean"}

    # numeric
    gt_num = try_parse_numeric(gt)
    if gt_num is not None:
        pred_num = try_parse_numeric(pred)
        if pred_num is not None:
            error = abs(pred_num - gt_num)
            tolerance = 0.5 if gt_num < 10 else (gt_num * 0.1)
            score = 1.0 if error <= tolerance else max(0.0, 1.0 - (error / (gt_num + 1e-6)))
            return {"score": score, "type": "numeric", "gt": gt_num, "pred": pred_num}
    
    # classification
    # gt 是非空、仅含字母 / 空格、长度 < 30 的字符串，则认为是分类问题
    is_categorical = re.match(r'^[A-Za-z\s]+$', gt) and len(gt) < 30
    if is_categorical:
        if gt.lower() == pred.lower():
            return {"score": 1.0, "type": "classification_exact"}
        if gt.lower() in pred.lower() or pred.lower() in gt.lower():
            return {"score": 0.8, "type": "classification_partial"}
    
    # text     
    common = set(pred.lower().split()) & set(gt.lower().split())
    score = len(common) / max(len(gt.split()), 1)
    return {"score": min(1.0, score), "type": "text_overlap"}


def evaluate_dataset():
    dataset_path = os.path.join(DATASET_ROOT, TARGET_FILE)
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    metadata_context = load_metadata(DATASET_ROOT)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # # 随机打乱数据集，并取前20条数据，用于调试
    # random.seed(42)
    # random.shuffle(dataset)
    # dataset = dataset[:10]
    print(f"Loaded {len(dataset)} items from {TARGET_FILE}")
    
    # save results to file
    results = []
    if resume:
        OUTPUT_FILE = os.path.join(OUTPUT_ROOT, resume_file)
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)['details']
                # remove results with API Error or empty prediction
                remain_results_ids = [r['id'] for r in results if r['prediction'] and r['prediction'] != 'API Error']
                results = [r for r in results if r['id'] in remain_results_ids]
                print(f"Loaded {len(results)} results from {resume_file}")
                for item in copy.deepcopy(dataset):
                    if item['Question_id'] in remain_results_ids:
                        # remove items that have already evaluated
                        dataset.remove(item)
                print(f"Remaining {len(dataset)} items to evaluate in dataset after resume.")
    else:
        OUTPUT_FILE = f"eval_result_{BASE_NAME}_{MODEL_NAME}_{TARGET_FILE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        OUTPUT_FILE = os.path.join(OUTPUT_ROOT, OUTPUT_FILE)
    
    # 1. 构建请求的参数
    def build_request(item, image_with_prefix: bool = True):
        question = item.get("Text", "")
        images_str = item.get("Image", "")
        stations_data = item.get("Stations", None)
        gt = str(item.get("Ground truth", ""))

        # system prompt
        system_prompt = "You are an expert AI for disaster assessment from satellite imagery and station data.\n"
        if metadata_context:
            system_prompt += metadata_context

        # user prompt
        # images data
        user_content = []
        img_msgs, paths = prepare_image_messages(images_str, RAW_DATA_BASE_PATH, TOTAL_MAX_SIZE, IMAGE_MAX_NUM)
        if image_with_prefix == True:
            for each_img_msg, each_path in zip(img_msgs, paths):
                path_split = each_path.split('/')
                day = path_split[-1].split('.')[0]
                band = path_split[-2]
                sensor = path_split[-3]
                user_content.append({"type": "text", "text": f"Image of Satellite Sensor: {sensor}, Band: {band}, Day: {day}"})
                user_content.append(each_img_msg)
        else:
            user_content.extend(img_msgs)
        
        # stations data
        if stations_data:
            st_text = format_station_data(stations_data)
            user_content.append({"type": "text", "text": f"Station Data Context:\n{st_text}"})
        
        # task instruction
        if 'assess the risk of an impending disaster' in question:
            # assess the risk of an impending disaster should answer "Yes" / "No"
            question += "Answer yes or no: yes means a disaster will occur, no means no disaster will occur."
        user_content.append({"type": "text", "text": f"Task Instruction: {question}"})

        # output requirement
        user_content.append({"type": "text", "text": 
            "\nOutput Requirement:\n"
            "Please analyze the inputs and provide the answer.\n"
            "You MUST output your final answer in a strict JSON format:\n"
            "```json\n"
            "{\"final_answer\": \"YOUR_ANSWER_HERE\", \"answer_type\": \"boolean/numeric/class/text\"}\n"
            "```\n"
            "For numeric answers, output only the number.\n"
            "For classification, output the class name."
        })

        req = {
            "item": item,
            "gt": gt,
            "system_prompt": system_prompt,
            "user_content": user_content
        }
        return req

    # 2. 定义单个请求的执行函数
    def do_call(item):
        req = build_request(item)
        raw_resp = ""
        raw_reasoning_content = ""
        item = req["item"]
        gt = req["gt"]
        system_prompt = req["system_prompt"]
        user_content = req["user_content"]

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_resp = resp.choices[0].message.content
            if hasattr(resp.choices[0].message, "reasoning_content"):
                raw_reasoning_content = resp.choices[0].message.reasoning_content
            extracted = extract_final_answer(raw_resp)
        except Exception as e:
            print(f"API Error: {e}")
            extracted = {"final_answer": "API Error"}

        score_info = qa_instance_score(gt, extracted)
        result_item = {
            "id": item.get("Question_id"),
            "task": item.get("Task"),
            "subtask": item.get("Subtask"),
            "ground_truth": gt,
            "prediction": extracted.get("final_answer"),
            "score": score_info["score"],
            "score_type": score_info.get("type"),
            "raw_response": raw_resp,
            "raw_reasoning_content": raw_reasoning_content
        }
        return (result_item, gt, extracted)

    # 3. 多线程并发执行所有请求（比如最多MAX_WORKERS个并发线程/任务）
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdm 需要外层包一层list使其能够预期获得总长度
        future_map = {executor.submit(do_call, item): idx for idx, item in enumerate(dataset)}
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_map), total=len(dataset))):
            result_item, gt, extracted = future.result()
            results.append(result_item)
            if i < 10:
                print(f"\n[Sample {i}] GT: {gt} | Pred: {extracted.get('final_answer')} | Score: {result_item['score']}")

    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0
    
    task_stats = {}
    for r in results:
        t = r.get("task", "Unknown")
        if t not in task_stats: task_stats[t] = []
        task_stats[t].append(r["score"])
        
    print("\n" + "="*40)
    print(f"Evaluation Complete. Overall Score: {avg_score:.4f}")
    print("Scores by Task:")
    for t, scores in task_stats.items():
        print(f"  - {t}: {sum(scores)/len(scores):.4f} (n={len(scores)})")
    print("="*40)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"summary": {"overall": avg_score, "by_task": {k: sum(v)/len(v) for k,v in task_stats.items()}}, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_dataset()