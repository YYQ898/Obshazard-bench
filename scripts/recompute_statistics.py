#!/usr/bin/env python3
"""
重新统计评测结果，添加灾害小类完整统计
功能：
1. 读取旧版评测结果文件
2. 从 details 提取每个样本的信息
3. 重新统计，生成新的包含灾害小类完整统计的JSON结构
4. 保存为新的文件
"""

import json
import os
import argparse
import re
from collections import defaultdict
from datetime import datetime


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


def recompute_statistics(eval_file, output_file=None):
    """
    重新统计评测结果
    
    Args:
        eval_file: 旧版评测结果文件路径
        output_file: 输出文件路径（可选）
    
    Returns:
        dict: 新的统计结果
    """
    # 读取旧版评测结果
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    old_summary = data.get("summary", {})
    details = data.get("details", [])
    
    print(f"\n读取文件: {eval_file}")
    print(f"总样本数: {len(details)}")
    
    # 统计字典
    # 灾害大类统计
    disaster_type_stats = {}
    
    # 灾害小类统计（新增完整层级）
    disaster_subtype_stats = {}
    disaster_subtype_task_stats = {}
    disaster_subtype_subtask_stats = {}
    disaster_subtype_tx_stats = {}
    
    # 任务统计
    task_stats = {}
    subtask_stats = {}
    tx_stats = {}
    
    # 遍历所有样本
    api_error_count = 0
    for r in details:
        # 过滤API Error样本
        prediction = r.get("prediction", "")
        if prediction == "API Error":
            api_error_count += 1
            continue
        
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
        
        score = r.get("score", 0)
        
        # 灾害大类统计
        if disaster_type:
            if disaster_type not in disaster_type_stats:
                disaster_type_stats[disaster_type] = []
            disaster_type_stats[disaster_type].append(score)
            
            # 灾害小类统计（新增完整层级）
            if disaster_type not in disaster_subtype_stats:
                disaster_subtype_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_stats[disaster_type]:
                disaster_subtype_stats[disaster_type][disaster_subtype] = []
            disaster_subtype_stats[disaster_type][disaster_subtype].append(score)
            
            # 灾害小类 -> 任务
            if disaster_type not in disaster_subtype_task_stats:
                disaster_subtype_task_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_task_stats[disaster_type]:
                disaster_subtype_task_stats[disaster_type][disaster_subtype] = {}
            if task not in disaster_subtype_task_stats[disaster_type][disaster_subtype]:
                disaster_subtype_task_stats[disaster_type][disaster_subtype][task] = []
            disaster_subtype_task_stats[disaster_type][disaster_subtype][task].append(score)
            
            # 灾害小类 -> 任务 -> 子任务
            if disaster_type not in disaster_subtype_subtask_stats:
                disaster_subtype_subtask_stats[disaster_type] = {}
            if disaster_subtype not in disaster_subtype_subtask_stats[disaster_type]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype] = {}
            if task not in disaster_subtype_subtask_stats[disaster_type][disaster_subtype]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task] = {}
            if subtask_name not in disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task]:
                disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task][subtask_name] = []
            disaster_subtype_subtask_stats[disaster_type][disaster_subtype][task][subtask_name].append(score)
            
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
            disaster_subtype_tx_stats[disaster_type][disaster_subtype][task][subtask_name][tx].append(score)
        
        # Task级
        if task not in task_stats:
            task_stats[task] = []
        task_stats[task].append(score)
        
        # Subtask级
        if task not in subtask_stats:
            subtask_stats[task] = {}
        if subtask_name not in subtask_stats[task]:
            subtask_stats[task][subtask_name] = []
        subtask_stats[task][subtask_name].append(score)
        
        # tx级
        if task not in tx_stats:
            tx_stats[task] = {}
        if subtask_name not in tx_stats[task]:
            tx_stats[task][subtask_name] = {}
        if tx not in tx_stats[task][subtask_name]:
            tx_stats[task][subtask_name][tx] = []
        tx_stats[task][subtask_name][tx].append(score)
    
    # 计算总体分数
    # 计算有效样本数（排除API Error）
    valid_samples = len(details) - api_error_count
    total_score = sum(r.get("score", 0) for r in details if r.get("prediction", "") != "API Error")
    avg_score = total_score / valid_samples if valid_samples else 0
    
    # 构建新的 JSON summary
    
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
    
    # 构建新的结果
    new_data = {
        "summary": {
            "overall": avg_score,
            "n": valid_samples,
            "total_samples": len(details),
            "api_errors": api_error_count,
            "by_disaster_type": summary_by_disaster_type,
            "by_task": summary_by_task
        },
        "details": details
    }
    
    # 保存结果
    if output_file is None:
        # 自动生成输出文件名
        base_name = os.path.basename(eval_file)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(os.path.dirname(eval_file), f"{name}_new{ext}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 新的统计结果已保存: {output_file}")
    
    # 打印统计摘要
    print("\n" + "="*70)
    print(f"总样本数: {len(details)}")
    print(f"API Error样本数: {api_error_count}")
    print(f"有效样本数: {valid_samples}")
    print(f"Overall Score: {avg_score:.4f}")
    print("="*70)
    
    print("\n=== By Disaster Type ===")
    for dt in sorted(disaster_type_stats.keys()):
        scores = disaster_type_stats[dt]
        dt_avg = sum(scores) / len(scores) if scores else 0
        print(f"{dt}: {dt_avg:.4f} (n={len(scores)})")
        
        if dt in disaster_subtype_stats:
            for dst in sorted(disaster_subtype_stats[dt].keys()):
                dst_scores = disaster_subtype_stats[dt][dst]
                dst_avg = sum(dst_scores) / len(dst_scores) if dst_scores else 0
                print(f"  - {dst}: {dst_avg:.4f} (n={len(dst_scores)})")
    
    print("\n=== By Task ===")
    for task in sorted(task_stats.keys()):
        scores = task_stats[task]
        task_avg = sum(scores) / len(scores) if scores else 0
        print(f"{task}: {task_avg:.4f} (n={len(scores)})")
    
    print("\n" + "="*70)
    
    return new_data


def main():
    parser = argparse.ArgumentParser(description="重新统计评测结果，添加灾害小类完整统计")
    parser.add_argument("--eval-file", required=True, help="旧版评测结果文件路径")
    parser.add_argument("--output", help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    recompute_statistics(args.eval_file, args.output)


if __name__ == "__main__":
    main()
