#!/usr/bin/env python3
"""
清理错误样本脚本（完整版）

功能：
1. 识别错误样本：Question_id包含"tmax"，但Time_Range不是真正的tmax
2. 从评测集JSON中删除错误样本
3. 从评测结果JSON中删除错误样本
4. 生成清理报告
"""

import json
import os
import argparse
from collections import defaultdict
from datetime import datetime


def get_tmax_from_question_id(question_id):
    """
    从Question_id提取时间步信息
    
    例如：
    - Impact/Total_Deaths/tmax/cold-wave1/2011-0105-MEX -> "tmax"
    - Impact/Total_Deaths/t1/cold-wave1/2011-0105-MEX -> "t1"
    """
    parts = question_id.split('/')
    if len(parts) >= 2:
        return parts[2]  # 返回时间步部分
    return None


def identify_error_samples(dataset_json):
    """
    识别错误样本
    
    错误样本：Question_id包含"tmax"，但Time_Range不是真正的tmax
    
    Returns:
        list: 错误样本列表
    """
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计每个Question_id的所有样本
    question_id_samples = defaultdict(list)
    for i, item in enumerate(data):
        question_id = item.get("Question_id", "")
        time_range = item.get("Time_Range", "")
        question_id_samples[question_id].append({
            "index": i,
            "time_range": time_range,
            "task": item.get("Task", ""),
            "subtask": item.get("Subtask", "")
        })
    
    # 识别错误样本
    error_samples = []
    
    for question_id, samples in question_id_samples.items():
        # 只检查包含"tmax"的Question_id
        if "tmax" not in question_id:
            continue
        
        # 如果只有一个样本，不是错误
        if len(samples) == 1:
            continue
        
        # 找出真正的tmax（最大的时间步）
        time_ranges = [s["time_range"] for s in samples]
        
        # 解析时间步
        def parse_time_step(t):
            if t == "tmax-1":
                return 9998
            elif t == "tmax":
                return 9999
            elif t.startswith("t") and t[1:].isdigit():
                return int(t[1:])
            else:
                return 0
        
        # 找出最大的时间步（真正的tmax）
        max_time_step = max(parse_time_step(t) for t in time_ranges)
        real_tmax = None
        for t in time_ranges:
            if parse_time_step(t) == max_time_step:
                real_tmax = t
                break
        
        # 标记错误样本（Time_Range不是真正的tmax）
        for sample in samples:
            if sample["time_range"] != real_tmax:
                error_samples.append({
                    "question_id": question_id,
                    "index": sample["index"],
                    "time_range": sample["time_range"],
                    "real_tmax": real_tmax,
                    "task": sample["task"],
                    "subtask": sample["subtask"]
                })
    
    return error_samples


def clean_dataset_json(dataset_json, error_samples, output_json=None):
    """
    从评测集JSON中删除错误样本
    
    Args:
        dataset_json: 原始JSON文件路径
        error_samples: 错误样本列表
        output_json: 输出JSON文件路径（可选）
    
    Returns:
        int: 删除的样本数
    """
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建错误索引集合
    error_indices = set(s["index"] for s in error_samples)
    
    # 过滤掉错误样本
    cleaned_data = [item for i, item in enumerate(data) if i not in error_indices]
    
    # 保存
    if output_json is None:
        output_json = dataset_json
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    return len(error_indices)


def clean_eval_result_json(eval_result_json, error_question_ids_with_time, output_json=None):
    """
    从评测结果JSON中删除错误样本
    
    Args:
        eval_result_json: 评测结果JSON文件路径
        error_question_ids_with_time: 错误样本的Question_id和Time_Range信息 {(question_id, time_range): True}
        output_json: 输出JSON文件路径（可选）
    
    Returns:
        int: 删除的样本数
    """
    with open(eval_result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤details
    original_details = data.get("details", [])
    cleaned_details = []
    removed_count = 0
    
    for item in original_details:
        question_id = item.get("id", "")
        subtask = item.get("subtask", "")
        
        # 从subtask中提取时间步
        # 例如："Total Deaths (tmax)" -> "tmax"
        import re
        m = re.search(r'\((tmax-1|tmax|t\d+)\)', subtask)
        time_step = m.group(1) if m else None
        
        # 检查是否是错误样本
        if (question_id, time_step) in error_question_ids_with_time:
            removed_count += 1
            continue
        
        cleaned_details.append(item)
    
    # 更新details
    data["details"] = cleaned_details
    
    # 保存
    if output_json is None:
        output_json = eval_result_json
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return removed_count


def main():
    parser = argparse.ArgumentParser(description="清理错误样本")
    parser.add_argument("--dataset-dir", required=True, help="评测集目录路径")
    parser.add_argument("--eval-results", nargs='+', help="评测结果JSON文件路径列表（可选）")
    parser.add_argument("--output-dir", help="输出目录路径（可选）")
    parser.add_argument("--dry-run", action="store_true", help="只识别不删除")
    
    args = parser.parse_args()
    
    # 找到所有灾害类型的JSON文件
    dataset_files = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file == "All.json":
                dataset_files.append(os.path.join(root, file))
    
    print(f"\n找到 {len(dataset_files)} 个评测集文件")
    
    # 识别所有错误样本
    all_error_samples = {}
    total_error_count = 0
    
    for dataset_file in dataset_files:
        disaster_type = os.path.basename(os.path.dirname(dataset_file)).replace("_json", "")
        print(f"\n检查 {disaster_type}...")
        
        error_samples = identify_error_samples(dataset_file)
        
        if error_samples:
            all_error_samples[disaster_type] = {
                "dataset_file": dataset_file,
                "error_samples": error_samples
            }
            total_error_count += len(error_samples)
            
            print(f"  发现 {len(error_samples)} 个错误样本:")
            for sample in error_samples[:5]:  # 只显示前5个
                print(f"    - {sample['question_id']}")
                print(f"      Time_Range: {sample['time_range']}, 真正的tmax: {sample['real_tmax']}")
            
            if len(error_samples) > 5:
                print(f"    ... 还有 {len(error_samples) - 5} 个错误样本")
        else:
            print(f"  没有发现错误样本")
    
    print(f"\n" + "="*70)
    print(f"总共发现 {total_error_count} 个错误样本")
    print("="*70)
    
    # 如果是dry-run模式，只识别不删除
    if args.dry_run:
        print("\n[DRY RUN] 只识别不删除")
        return
    
    # 删除错误样本
    print("\n开始删除错误样本...")
    
    # 构建错误样本的Question_id和Time_Range集合
    error_question_ids_with_time = set()
    
    # 从评测集JSON中删除
    for disaster_type, info in all_error_samples.items():
        dataset_file = info["dataset_file"]
        error_samples = info["error_samples"]
        
        # 收集错误样本的Question_id和Time_Range
        for sample in error_samples:
            error_question_ids_with_time.add((sample["question_id"], sample["time_range"]))
        
        removed_count = clean_dataset_json(dataset_file, error_samples)
        print(f"  {disaster_type}: 从评测集删除了 {removed_count} 个错误样本")
    
    # 从评测结果JSON中删除
    if args.eval_results:
        print("\n从评测结果中删除错误样本...")
        for eval_result_file in args.eval_results:
            if os.path.exists(eval_result_file):
                removed_count = clean_eval_result_json(eval_result_file, error_question_ids_with_time)
                print(f"  {os.path.basename(eval_result_file)}: 删除了 {removed_count} 个错误样本")
    
    print("\n[OK] 清理完成")


if __name__ == "__main__":
    main()
