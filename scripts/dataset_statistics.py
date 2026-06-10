#!/usr/bin/env python3
"""
统计数据集信息
"""

import os
import json
from collections import Counter
from pathlib import Path

def count_samples(json_file):
    """统计单个 JSON 文件的样本数量"""
    if not os.path.exists(json_file):
        return 0

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return len(data)

def analyze_dataset(dataset_root):
    """分析整个数据集"""
    dataset_root = Path(dataset_root)

    # 灾害类型映射
    disaster_types = {
        'cold-wave_json': 'cold-wave',
        'heat-wave_json': 'heat-wave',
        'earthquake_json': 'earthquake',
        'flood_json': 'flood',
        'mass movement (wet)_json': 'mass movement (wet)',
        'storm_json': 'storm',
        'volcanic activity_json': 'volcanic activity',
        'wildfire_json': 'wildfire'
    }

    results = {}

    for json_dir, disaster_name in disaster_types.items():
        json_path = dataset_root / json_dir / 'All.json'
        sample_count = count_samples(json_path)

        if sample_count > 0:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 统计事件ID
            event_ids = set()
            task_dist = Counter()
            subtask_dist = Counter()

            for item in data:
                event_id = item.get('Event ID', '')
                if event_id:
                    event_ids.add(event_id)

                task = item.get('Task', '')
                if task:
                    task_dist[task] += 1

                subtask = item.get('Subtask', '')
                if subtask:
                    subtask_dist[subtask] += 1

            results[disaster_name] = {
                'samples': sample_count,
                'events': len(event_ids),
                'tasks': dict(task_dist),
                'subtasks': dict(subtask_dist)
            }

    return results

def print_statistics(results):
    """打印统计结果"""
    print("=" * 80)
    print("ObsCrisis-Bench 数据集统计")
    print("=" * 80)
    print()

    total_samples = 0
    total_events = 0

    # 表头
    print(f"{'灾害类型':<25} {'事件数':<10} {'样本数':<10}")
    print("-" * 45)

    for disaster_name, stats in sorted(results.items()):
        events = stats['events']
        samples = stats['samples']
        total_events += events
        total_samples += samples
        print(f"{disaster_name:<25} {events:<10} {samples:<10}")

    print("-" * 45)
    print(f"{'总计':<25} {total_events:<10} {total_samples:<10}")
    print()

    # 任务类型统计
    print("=" * 80)
    print("任务类型分布")
    print("=" * 80)

    all_tasks = Counter()
    for disaster_name, stats in results.items():
        for task, count in stats['tasks'].items():
            all_tasks[task] += count

    print(f"\n{'任务类型':<35} {'样本数':<10} {'占比':<10}")
    print("-" * 55)
    for task, count in sorted(all_tasks.items()):
        percentage = count / total_samples * 100
        print(f"{task:<35} {count:<10} {percentage:.2f}%")
    print()

    # 子任务统计（前20个）
    print("=" * 80)
    print("子任务分布（Top 20）")
    print("=" * 80)

    all_subtasks = Counter()
    for disaster_name, stats in results.items():
        for subtask, count in stats['subtasks'].items():
            all_subtasks[subtask] += count

    print(f"\n{'子任务':<50} {'样本数':<10}")
    print("-" * 60)
    for subtask, count in all_subtasks.most_common(20):
        print(f"{subtask:<50} {count:<10}")
    print()

    # Markdown 格式输出
    print("=" * 80)
    print("Markdown 表格格式")
    print("=" * 80)
    print()
    print("### 灾害类别统计")
    print()
    print("| 类别 | 事件数 | VQA样本数 |")
    print("|------|--------|-----------|")
    for disaster_name, stats in sorted(results.items()):
        print(f"| {disaster_name} | {stats['events']} | {stats['samples']} |")
    print(f"| **总计** | **{total_events}** | **{total_samples}** |")
    print()

    print("### 任务类型统计")
    print()
    print("| 任务类型 | 样本数 | 占比 |")
    print("|----------|--------|------|")
    for task, count in sorted(all_tasks.items()):
        percentage = count / total_samples * 100
        print(f"| {task} | {count} | {percentage:.2f}% |")
    print()

if __name__ == "__main__":
    dataset_root = Path(__file__).parent.parent / "test"
    results = analyze_dataset(dataset_root)
    print_statistics(results)
