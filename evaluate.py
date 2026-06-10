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
    image_max_num: int,
    sample_method: str = "evenly"
) -> List[Dict[str, Any]]:
    """
    请求输入要求：图片大小总和不可以大于total_max_size，图片数量最多image_max_num
    """
    if not image_rel_paths:
        return []
    
    paths = [p.strip() for p in image_rel_paths.split(',') if p.strip()]
    messages = []
    
    # 如果图片数量超过最大数量，则进行采样
    if len(paths) > image_max_num:
        original_num = len(paths)
        if sample_method == "first":
            paths = paths[:image_max_num]
            print(f"Warning: Image number {original_num} exceeds max_num ({image_max_num}), only use the first {image_max_num} images")
        elif sample_method == "evenly":
            if image_max_num <= 1:
                paths = [paths[0]]
            else:
                step = (len(paths) - 1) / (image_max_num - 1)
                indices = [round(i * step) for i in range(image_max_num)]
                paths = [paths[i] for i in indices]
                print(
                    f"Warning: Image number {original_num} exceeds max_num ({image_max_num}), "
                    f"evenly sample to {len(paths)} images"
                )
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
    
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
    # dataset = dataset[:20]
    print(f"Loaded {len(dataset)} items from {TARGET_FILE}")
    
    # save results to file
    results = []
    if resume:
        OUTPUT_FILE = os.path.join(OUTPUT_ROOT, resume_file)
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                results = json.load(f)['details']
                # remove results with API Error or empty prediction
                remain_results_ids = [r['id'] for r in results if str(r['prediction']).strip() and str(r['prediction']).strip() != 'API Error']
                results = [r for r in results if r['id'] in remain_results_ids]
                print(f"Loaded {len(results)} results from {resume_file}")
                for item in copy.deepcopy(dataset):
                    if item['Question_id'] in remain_results_ids:
                        # remove items that have already evaluated
                        dataset.remove(item)
                print(f"Remaining {len(dataset)} items to evaluate in dataset after resume.")
    else:
        safe_target = TARGET_FILE.replace('/', '_').replace('\\', '_')
        OUTPUT_FILE = f"eval_result_{BASE_NAME}_{MODEL_NAME}_{safe_target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        OUTPUT_FILE = os.path.join(OUTPUT_ROOT, OUTPUT_FILE)
    
    # 1. 构建请求的参数
    def build_request(item, image_with_prefix: bool = True):
        question = item.get("Text", "")
        images_str = item.get("Image", "")
        stations_data = item.get("Stations", None)
        gt = str(item.get("Ground truth", ""))

        # system prompt
        system_prompt = """
You are a professional meteorological disaster analysis expert, specializing in analyzing disaster-related issues using multispectral satellite imagery data.

## Data Composition Description

### Satellite Sensors and Band Configuration

You will receive multi-band image data from three satellite sensors:

#### 1. MHS (Microwave Humidity Sounder)
| Band No. | Center Frequency (GHz) | Bandwidth (MHz) | Polarization | NEΔT (K) |
|:-------:|:---------------------:|:---------------:|:-----------:|:--------:|
| 0 | 89.0 | 2800 | QV | 0.22 |
| 1 | 157.0 | 2800 | QV | 0.38 |
| 2 | 183.31 ± 1.0 | 1000 | QH | 0.57 |
| 3 | 183.31 ± 3.0 | 2000 | QH | 0.42 |
| 4 | 190.311 | 2000 | QV | 0.45 |
- **Frequency Range**: 89.0 GHz - 190.3 GHz
- **Number of Bands**: 5 (No. 0-4)
- **Primary Uses**: Atmospheric humidity profile sounding, precipitation and water vapor distribution monitoring, storm intensity assessment

#### 2. HIRS (High Resolution Infrared Radiation Sounder)
**Infrared Band Characteristics (Wavelength Range: 0.69 μm - 14.95 μm)**
| Band No. | Wavelength | Wave Number (cm⁻¹) |
|:-------:|:----------:|:------------------:|
| 0 | 14.95 μm | 669 |
| 1 | 14.71 μm | 680 |
| 2 | 14.49 μm | 690 |
| 3 | 14.22 μm | 703 |
| 4 | 13.97 μm | 716 |
| 5 | 13.64 μm | 733 |
| 6 | 13.35 μm | 749 |
| 7 | 12.47 μm | 802 |
| 8 | 11.11 μm | 900 |
| 9 | 9.71 μm | 1030 |
| 10 | 7.33 μm | 1364 |
| 11 | 6.52 μm | 1534 |
| 12 | 4.57 μm | 2188 |
| 13 | 4.52 μm | 2210 |
| 14 | 4.47 μm | 2237 |
| 15 | 4.45 μm | 2247 |
| 16 | 4.13 μm | 2420 |
| 17 | 4.00 μm | 2500 |
| 18 | 3.76 μm | 2660 |
| 19 | 0.69 μm | N/A |
- **Number of Bands**: 20 (No. 0-19)
- **Primary Uses**: High-resolution infrared radiation sounding, cloud top temperature identification, thermal anomaly monitoring, volcanic activity and wildfire detection

#### 3. AMSU-A (Advanced Microwave Sounding Unit-A)
| Band No. | Center Frequency (GHz) | Bandwidth (MHz) | Polarization | NEΔT (K) |
|:-------:|:---------------------:|:---------------:|:-----------:|:--------:|
| 0 | 23.800 | 270 | QV | 0.30 |
| 1 | 31.400 | 180 | QV | 0.30 |
| 2 | 50.300 | 180 | QV | 0.40 |
| 3 | 52.800 | 400 | QV | 0.25 |
| 4 | 53.596 ± 0.115 | 170 | QH | 0.25 |
| 5 | 54.400 | 400 | QH | 0.25 |
| 6 | 54.940 | 400 | QV | 0.25 |
| 7 | 55.500 | 330 | QH | 0.25 |
| 8 | 57.290344 | 330 | QH | 0.25 |
| 9 | f0 ± 0.217 | 78 | QH | 0.40 |
| 10 | f0 ± 0.3222 ± 0.048 | 36 | QH | 0.40 |
| 11 | f0 ± 0.3222 ± 0.022 | 16 | QH | 0.60 |
| 12 | f0 ± 0.3222 ± 0.010 | 8 | QH | 0.80 |
| 13 | f0 ± 0.3222 ± 0.0045 | 3 | QH | 1.20 |
| 14 | 89.000 ± 1.0 | 1000 | QV | 0.50 |
- **Frequency Range**: 23.8 GHz - 89.0 GHz
- **Number of Bands**: 15 (No. 0-14)
- **Primary Uses**: Atmospheric temperature and humidity vertical profile sounding, cloud structure monitoring, precipitation intensity assessment, atmospheric stability analysis

### Data File Structure

Each event's data is organized as follows:
```
EventID/
├── AMSU-A/
│   ├── 0/         (Band 0 directory, contains all time-point images for this band)
│   │   ├── t1.png
│   │   ├── t8.png
│   │   └── ...    (Multiple time-point images)
│   ├── 1/         (Band 1 directory)
│   └── ...        (Total 15 band directories, No. 0-14)
├── HIRS/
│   ├── 0/         (Band 0 directory, contains all time-point images for this band)
│   ├── 1/         (Band 1 directory)
│   └── ...        (Total 20 band directories, No. 0-19)
└── MHS/
    ├── 0/         (Band 0 directory, contains all time-point images for this band)
    ├── 1/         (Band 1 directory)
    └── ...        (Total 5 band directories, No. 0-4)
```

### Data Characteristics

- Each band directory (e.g., MHS/0/) contains multiple time-point observation images for that band
- The numbers in image filenames (e.g., t1, t8, t12) only indicate the chronological order of the images
- Band data from different sensors provides complementary observation information
- Meteorological station data (such as temperature, humidity, pressure, etc.) may be provided as auxiliary information

Please analyze and answer questions based on the provided satellite image data.
"""
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
            raw_resp = resp.choices[0].message.content or ""
            if hasattr(resp.choices[0].message, "reasoning_content") and resp.choices[0].message.reasoning_content:
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

    # 四级分组统计: DisasterType -> DisasterSubtype -> Task -> Subtask -> tx
    import re
    
    # 灾害小类到大类的映射
    SUBTYPE_TO_TYPE = {
        "ground movement": "earthquake", "tsunami": "earthquake",
        "cold-wave1": "cold-wave", "heat-wave1": "heat-wave",
        "coastal flood": "flood", "flash flood": "flood", "general flood": "flood", "riverine flood": "flood",
        "landslide wet": "mass movement (wet)", "mudslide": "mass movement (wet)",
        "hail": "storm", "tornado": "storm", "tropical cyclone": "storm", "extra-tropical cyclone": "storm",
        "convective storm": "storm", "general storm": "storm", "storm surge": "storm", "lightning": "storm",
        "sandstorm": "storm", "severe weather": "storm", "wind": "storm",
        "ash fall": "volcanic activity", "general activity": "volcanic activity", "lava flow": "volcanic activity",
        "pyroclastic flow": "volcanic activity", "lahar": "volcanic activity",
        "forest fire": "wildfire", "general wildfire": "wildfire", "land fire": "wildfire",
    }
    
    def extract_disaster_info(question_id):
        """从 Question_id 提取灾害大类和小类"""
        parts = question_id.split('/')
        if len(parts) >= 4:
            disaster_subtype = parts[3]
            disaster_type = SUBTYPE_TO_TYPE.get(disaster_subtype, "unknown")
            return disaster_type, disaster_subtype
        return None, None
    
    # 统计字典
    # 灾害大类统计
    disaster_type_stats = {}  # disaster_type -> [scores]
    
    # 灾害小类统计（新增完整层级）
    disaster_subtype_stats = {}  # disaster_type -> disaster_subtype -> [scores]
    disaster_subtype_task_stats = {}  # disaster_type -> disaster_subtype -> task -> [scores]
    disaster_subtype_subtask_stats = {}  # disaster_type -> disaster_subtype -> task -> subtask -> [scores]
    disaster_subtype_tx_stats = {}  # disaster_type -> disaster_subtype -> task -> subtask -> tx -> [scores]
    
    # 任务统计
    task_stats = {}       # task -> [scores]
    subtask_stats = {}    # task -> subtask_name -> [scores]
    tx_stats = {}         # task -> subtask_name -> tx -> [scores]

    for r in results:
        # 提取灾害信息
        question_id = r.get("id", "")
        disaster_type, disaster_subtype = extract_disaster_info(question_id)
        
        task = r.get("task", "Unknown")
        subtask_raw = r.get("subtask", "")
        
        # 解析Subtask: "Risk Detection (t1)" -> name="Risk Detection", tx="t1"
        m = re.match(r'(.+?)\s*\((tmax-1|tmax|t\d+)\)', subtask_raw)
        if m:
            subtask_name = m.group(1).strip()
            tx = m.group(2)
        else:
            subtask_name = subtask_raw.strip()
            tx = "overall"
        
        # 灾害大类统计
        if disaster_type:
            if disaster_type not in disaster_type_stats:
                disaster_type_stats[disaster_type] = []
            disaster_type_stats[disaster_type].append(r["score"])
            
            # 灾害小类统计（新增完整层级）
            if disaster_type not in disaster_subtype_stats:
                disaster_subtype_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_stats[disaster_type]:
                disaster_subtype_stats[disaster_type][disaster_subtype] = []
            disaster_subtype_stats[disaster_type][disaster_subtype].append(r["score"])
            
            # 灾害小类 -> 任务
            if disaster_type not in disaster_subtype_task_stats:
                disaster_subtype_task_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_task_stats[disaster_type]:
                disaster_subtype_task_stats[disaster_type][disaster_subtype] = {}
            if task not in disaster_subtype_task_stats[disaster_type][disaster_subtype]:
                disaster_subtype_task_stats[disaster_type][disaster_subtype][task] = []
            disaster_subtype_task_stats[disaster_type][disaster_subtype][task].append(r["score"])
            
            # 灾害小类 -> 任务 -> 子任务
            if disaster_type not in disaster_subtype_subtask_stats:
                disaster_subtype_subtask_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_subtask_stats[disaster_type]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype] = {}
            if task not in disaster_subtype_subtask_stats[disaster_type][disaster_subtype]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task] = {}
            if subtask_name not in disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task][subtask_name] = []
            disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task][subtask_name].append(r["score"])
            
            # 灾害小类 -> 任务 -> 子任务 -> 时间步
            if disaster_type not in disaster_subtype_tx_stats:
                disaster_subtype_tx_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_tx_stats[disaster_type]:
                disaster_subtype_tx_stats[disaster_type][disaster_subtype] = {}
            if task not in disaster_subtype_tx_stats[disaster_type][disaster_subtype]:
                disaster_subtype_tx_stats[disaster_type][disaster_subtype][task] = {}
            if subtask_name not in disaster_subtype_tx_stats[disaster_type][disaster_subtype][task]:
                disaster_subtype_tx_stats[disaster_type][disaster_subtype][task][subtask_name] = {}
            if tx not in disaster_subtype_tx_stats[disaster_type][disaster_subtype][task][subtask_name]:
                disaster_subtype_tx_stats[disaster_type][disaster_subtype][task][subtask_name][tx] = []
            disaster_subtype_tx_stats[disaster_type][disaster_subtype][task][subtask_name][tx].append(r["score"])

        # Task级
        if task not in task_stats:
            task_stats[task] = []
        task_stats[task].append(r["score"])

        # Subtask级
        if task not in subtask_stats:
            subtask_stats[task] = {}
        if subtask_name not in subtask_stats[task]:
            subtask_stats[task][subtask_name] = []
        subtask_stats[task][subtask_name].append(r["score"])

        # tx级
        if task not in tx_stats:
            tx_stats[task] = {}
        if subtask_name not in tx_stats[task]:
            tx_stats[task][subtask_name] = {}
        if tx not in tx_stats[task][subtask_name]:
            tx_stats[task][subtask_name][tx] = []
        tx_stats[task][subtask_name][tx].append(r["score"])

    # 打印多层级统计
    print("\n" + "="*70)
    print(f"Evaluation Complete. Overall Score: {avg_score:.4f} (n={len(results)})")
    print("="*70)
    
    # 打印灾害大类和小类统计（新增详细统计）
    if disaster_type_stats:
        print("\n=== By Disaster Type ===")
        for dt in sorted(disaster_type_stats.keys()):
            scores = disaster_type_stats[dt]
            dt_avg = sum(scores) / len(scores) if scores else 0
            print(f"\n{dt}: {dt_avg:.4f} (n={len(scores)})")
            
            # 打印灾害小类详细统计
            if dt in disaster_subtype_stats:
                for dst in sorted(disaster_subtype_stats[dt].keys()):
                    dst_scores = disaster_subtype_stats[dt][dst]
                    dst_avg = sum(dst_scores) / len(dst_scores) if dst_scores else 0
                    print(f"\n  === {dst} (n={len(dst_scores)}) ===")
                    print(f"  Overall: {dst_avg:.4f}")
                    
                    # 打印灾害小类的任务统计
                    if dt in disaster_subtype_task_stats and dst in disaster_subtype_task_stats[dt]:
                        for task in sorted(disaster_subtype_task_stats[dt][dst].keys()):
                            task_scores = disaster_subtype_task_stats[dt][dst][task]
                            task_avg = sum(task_scores) / len(task_scores) if task_scores else 0
                            print(f"\n  {task}: {task_avg:.4f} (n={len(task_scores)})")
                            
                            # 打印子任务
                            if dt in disaster_subtype_subtask_stats and dst in disaster_subtype_subtask_stats[dt]:
                                if task in disaster_subtype_subtask_stats[dt][dst]:
                                    for subtask_name in sorted(disaster_subtype_subtask_stats[dt][dst][task].keys()):
                                        st_scores = disaster_subtype_subtask_stats[dt][dst][task][subtask_name]
                                        st_avg = sum(st_scores) / len(st_scores) if st_scores else 0
                                        print(f"    {subtask_name}: {st_avg:.4f} (n={len(st_scores)})")
                                        
                                        # 打印时间步
                                        if dt in disaster_subtype_tx_stats and dst in disaster_subtype_tx_stats[dt]:
                                            if task in disaster_subtype_tx_stats[dt][dst] and subtask_name in disaster_subtype_tx_stats[dt][dst][task]:
                                                # 排序: t1, t8, t12, ..., tmax-1, tmax
                                                def tx_sort_key(x):
                                                    if x == 'tmax-1': return 998
                                                    if x == 'tmax': return 999
                                                    if x.startswith('t') and x[1:].isdigit(): return int(x[1:])
                                                    return 500
                                                sorted_tx = sorted(disaster_subtype_tx_stats[dt][dst][task][subtask_name].keys(), key=tx_sort_key)
                                                for tx in sorted_tx:
                                                    tx_scores = disaster_subtype_tx_stats[dt][dst][task][subtask_name][tx]
                                                    tx_avg = sum(tx_scores) / len(tx_scores) if tx_scores else 0
                                                    print(f"      {tx}: {tx_avg:.4f} (n={len(tx_scores)})")

    # 打印任务统计
    print("\n\n=== By Task ===")
    for task in sorted(task_stats.keys()):
        scores = task_stats[task]
        task_avg = sum(scores) / len(scores) if scores else 0
        print(f"\n{task}: {task_avg:.4f} (n={len(scores)})")

        if task in subtask_stats:
            for subtask_name in sorted(subtask_stats[task].keys()):
                st_scores = subtask_stats[task][subtask_name]
                st_avg = sum(st_scores) / len(st_scores) if st_scores else 0
                print(f"  {subtask_name}: {st_avg:.4f} (n={len(st_scores)})")

                if task in tx_stats and subtask_name in tx_stats[task]:
                    # 排序: t1, t8, t12, ..., tmax-1, tmax
                    def tx_sort_key(x):
                        if x == 'tmax-1': return 998
                        if x == 'tmax': return 999
                        if x.startswith('t') and x[1:].isdigit(): return int(x[1:])
                        return 500
                    sorted_tx = sorted(tx_stats[task][subtask_name].keys(), key=tx_sort_key)
                    for tx in sorted_tx:
                        tx_scores = tx_stats[task][subtask_name][tx]
                        tx_avg = sum(tx_scores) / len(tx_scores) if tx_scores else 0
                        print(f"    {tx}: {tx_avg:.4f} (n={len(tx_scores)})")

    print("\n" + "="*70)

    # 构建嵌套的JSON summary
    
    # 灾害大类和小类统计（新增完整层级）
    summary_by_disaster_type = {}
    for dt in sorted(disaster_type_stats.keys()):
        scores = disaster_type_stats[dt]
        dt_entry = {
            "overall": sum(scores) / len(scores) if scores else 0,
            "n": len(scores),
            "by_subtype": {}
        }
        if dt in disaster_subtype_stats:
            for dst in sorted(disaster_subtype_stats[dt].keys()):
                dst_scores = disaster_subtype_stats[dt][dst]
                dst_entry = {
                    "overall": sum(dst_scores) / len(dst_scores) if dst_scores else 0,
                    "n": len(dst_scores),
                    "by_task": {}
                }
                
                # 灾害小类的任务统计
                if dt in disaster_subtype_task_stats and dst in disaster_subtype_task_stats[dt]:
                    for task in sorted(disaster_subtype_task_stats[dt][dst].keys()):
                        task_scores = disaster_subtype_task_stats[dt][dst][task]
                        task_entry = {
                            "overall": sum(task_scores) / len(task_scores) if task_scores else 0,
                            "n": len(task_scores),
                            "by_subtask": {}
                        }
                        
                        # 灾害小类的子任务统计
                        if dt in disaster_subtype_subtask_stats and dst in disaster_subtype_subtask_stats[dt]:
                            if task in disaster_subtype_subtask_stats[dt][dst]:
                                for subtask_name in sorted(disaster_subtype_subtask_stats[dt][dst][task].keys()):
                                    st_scores = disaster_subtype_subtask_stats[dt][dst][task][subtask_name]
                                    subtask_entry = {
                                        "overall": sum(st_scores) / len(st_scores) if st_scores else 0,
                                        "n": len(st_scores),
                                        "by_timestep": {}
                                    }
                                    
                                    # 灾害小类的时间步统计
                                    if dt in disaster_subtype_tx_stats and dst in disaster_subtype_tx_stats[dt]:
                                        if task in disaster_subtype_tx_stats[dt][dst] and subtask_name in disaster_subtype_tx_stats[dt][dst][task]:
                                            def tx_sort_key(x):
                                                if x == 'tmax-1': return 998
                                                if x == 'tmax': return 999
                                                if x.startswith('t') and x[1:].isdigit(): return int(x[1:])
                                                return 500
                                            sorted_tx = sorted(disaster_subtype_tx_stats[dt][dst][task][subtask_name].keys(), key=tx_sort_key)
                                            for tx in sorted_tx:
                                                tx_scores = disaster_subtype_tx_stats[dt][dst][task][subtask_name][tx]
                                                subtask_entry["by_timestep"][tx] = {
                                                    "score": sum(tx_scores) / len(tx_scores) if tx_scores else 0,
                                                    "n": len(tx_scores)
                                                }
                                    task_entry["by_subtask"][subtask_name] = subtask_entry
                        dst_entry["by_task"][task] = task_entry
                dt_entry["by_subtype"][dst] = dst_entry
        summary_by_disaster_type[dt] = dt_entry
    
    # 任务统计
    summary_by_task = {}
    for task in sorted(task_stats.keys()):
        scores = task_stats[task]
        task_entry = {
            "overall": sum(scores) / len(scores) if scores else 0,
            "n": len(scores),
            "by_subtask": {}
        }
        if task in subtask_stats:
            for subtask_name in sorted(subtask_stats[task].keys()):
                st_scores = subtask_stats[task][subtask_name]
                subtask_entry = {
                    "overall": sum(st_scores) / len(st_scores) if st_scores else 0,
                    "n": len(st_scores),
                    "by_timestep": {}
                }
                if task in tx_stats and subtask_name in tx_stats[task]:
                    def tx_sort_key(x):
                        if x == 'tmax-1': return 998
                        if x == 'tmax': return 999
                        if x.startswith('t') and x[1:].isdigit(): return int(x[1:])
                        return 500
                    sorted_tx = sorted(tx_stats[task][subtask_name].keys(), key=tx_sort_key)
                    for tx in sorted_tx:
                        tx_scores = tx_stats[task][subtask_name][tx]
                        subtask_entry["by_timestep"][tx] = {
                            "score": sum(tx_scores) / len(tx_scores) if tx_scores else 0,
                            "n": len(tx_scores)
                        }
                task_entry["by_subtask"][subtask_name] = subtask_entry
        summary_by_task[task] = task_entry

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "overall": avg_score, 
                "n": len(results), 
                "by_disaster_type": summary_by_disaster_type,
                "by_task": summary_by_task
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_dataset()