# Extreme Events — 极端天气事件多模态评测框架

本项目是一个针对大型视觉语言模型（LVM）的评测框架，用于评估模型在**极端天气事件**（地震、风暴等）分析任务上的表现。评测输入包括卫星多光谱图像与气象站传感器数据，输出涵盖灾害风险预判、强度预测、类别识别等多类任务的得分。

---

## 功能特性

- 支持多模态输入：卫星多波段图像 + 气象站传感器时序数据
- 支持多种大模型评测：GPT-4o、Gemini 3 Pro、Qwen 3.5 等
- 自动图像压缩，适配不同模型的上下文大小限制
- 多线程并发请求，评测效率高
- 支持断点续评（`--resume`）
- 按任务类型分类统计得分

---

## 环境要求

- Python 3.8+
- 可访问的 OpenAI 兼容 API 端点

```bash
pip install -r requirements.txt
```

依赖项：
- `openai==2.21.0`
- `Pillow==12.1.1`

---

## 数据集结构

数据集根目录下应包含以下文件：

```
<dataset_root>/
├── Image_Only.json           # 仅含图像的评测集
├── Image_With_Station.json   # 图像 + 气象站数据的评测集
└── dataset_metadata.json     # （可选）卫星传感器及站点字段定义
```

每条数据样本的字段：

| 字段 | 说明 |
|------|------|
| `Question_id` | 唯一标识符 |
| `Task` | 任务大类（如 earthquake、storm） |
| `Subtask` | 子任务类型 |
| `Text` | 问题文本 |
| `Image` | 逗号分隔的图像相对路径列表 |
| `Stations` | 气象站传感器数据（JSON 对象，可选） |
| `Ground truth` | 标准答案 |

---

## 快速开始

**配置 API 凭据：**

```bash
export OPENAI_BASE_URL=<your_api_base_url>
export OPENAI_API_KEY=<your_api_key>
```

**运行评测（使用 run.sh 中预配置的参数）：**

```bash
bash run.sh
```

**直接运行 evaluate.py：**

```bash
python evaluate.py \
  --dataset-root /path/to/dataset \
  --raw-data-base-path /path/to/images \
  --target-file Image_With_Station.json \
  --model-name gpt-4o \
  --max-workers 64 \
  --image-max-num 500 \
  --total-max-size 47185920 \
  --temperature 0.1 \
  --max-tokens 300
```

**恢复上次中断的评测：**

```bash
python evaluate.py <其他参数> \
  --resume \
  --resume-file output/eval_result_<timestamp>.json
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-root` | 必填 | 数据集 JSON 文件所在目录 |
| `--raw-data-base-path` | 必填 | 卫星图像文件的根目录 |
| `--target-file` | `Image_Only.json` | 评测集文件名 |
| `--model-name` | `gpt-4o` | 模型名称 |
| `--max-workers` | `64` | 并发线程数 |
| `--image-max-num` | `500` | 单次请求最多图像数 |
| `--total-max-size` | `47185920`（45 MB） | 单次请求图像总大小上限（字节） |
| `--temperature` | `0.1` | 模型温度参数 |
| `--max-tokens` | `300` | 最大输出 token 数 |
| `--resume` | — | 开启断点续评模式 |
| `--resume-file` | — | 续评时读取的历史结果文件 |

---

## 支持的模型

在 `run.sh` 中预定义了以下模型的推荐参数：

| 模型 | 最大图像数 | 温度 | 备注 |
|------|-----------|------|------|
| `gpt-4o` | 500 | 0.1 | OpenAI 官方 API |
| `gemini-3-pro-preview-thinking` | 600 | 1.0 | 通过 OpenAI 兼容适配器，支持 `reasoning_content` |
| `qwen3.5-397b-a17b` | 250 | 0.6 | 阿里云 API，关闭 thinking 模式 |

所有模型均通过 OpenAI 兼容 API 访问，可根据需要扩展其他模型。

---

## 输出格式

结果保存至 `output/eval_result_<dataset>_<model>_<file>_<timestamp>.json`：

```json
{
  "summary": {
    "overall": 0.85,
    "by_task": {
      "earthquake": 0.90,
      "storm": 0.80
    }
  },
  "details": [
    {
      "id": "Q001",
      "task": "storm",
      "subtask": "intensity",
      "ground_truth": "Category 3",
      "prediction": "Category 3",
      "score": 1.0,
      "score_type": "classification_exact",
      "raw_response": "...",
      "raw_reasoning_content": "..."
    }
  ]
}
```

### 评分规则

| 类型 | 判断条件 | 规则 |
|------|---------|------|
| 布尔型 | 答案为 yes/no | 精确匹配，1.0 或 0.0 |
| 数值型 | 答案可解析为数字 | 误差 ≤10% 得 1.0，>20% 得 0.0，中间线性衰减 |
| 分类型 | 纯字母字符串且长度 < 30 | 精确匹配 1.0，包含匹配 0.8 |
| 文本型 | 其他情况 | 基于词集合的 Jaccard 重叠分 |

---

## 工具脚本

```bash
# 校验输出结果文件的完整性
python scripts/check.py

# 分析数据集统计信息（任务分布、图像数量等）
python scripts/data_analyse.py
```
