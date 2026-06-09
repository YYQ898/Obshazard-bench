# ObsCrisis-Bench — Extreme Weather Event Multimodal Evaluation Framework

ObsCrisis-Bench is a benchmark for evaluating large vision-language models on **extreme weather event** analysis tasks. The evaluation input includes satellite multispectral imagery and weather station sensor data, covering disaster risk assessment, intensity prediction, type classification, and more.

---

## Features

- Multimodal input: satellite multi-band imagery + weather station time-series data
- Support for multiple LVMs: GPT-4o, Gemini 3 Pro, Qwen 3.5, etc.
- Automatic image compression to fit different model context size limits
- Multi-threaded concurrent requests for efficient evaluation
- Resume from interrupted evaluation (`--resume`)
- Hierarchical scoring: **Task → Subtask → Timestep (tx)** level statistics

---

## Dataset

The benchmark dataset is available on HuggingFace: [ObsCrisis-Bench](https://huggingface.co/datasets/YYQ898/ObsCrisis-Bench)

### Dataset Structure

```
ObsCrisis-Bench/
├── cold-wave/                          # Satellite imagery
│   └── cold-wave1/
│       └── <event_id>/
│           ├── AMSU-A/<band>/t1.png ... tN.png
│           ├── HIRS/<band>/t1.png ... tN.png
│           └── MHS/<band>/t1.png ... tN.png
├── cold-wave_json/
│   └── All.json                        # VQA records
├── heat-wave/
├── heat-wave_json/
│   └── All.json
├── earthquake/
├── earthquake_json/
│   └── All.json
├── flood/
├── flood_json/
│   └── All.json
├── mass movement (wet)/
├── mass movement (wet)_json/
│   └── All.json
├── storm/
├── storm_json/
│   └── All.json
├── volcanic activity/
├── volcanic activity_json/
│   └── All.json
├── wildfire/
└── wildfire_json/
    └── All.json
```

### Disaster Categories

| Category | Events | VQA Samples | Subtypes |
|----------|--------|-------------|----------|
| cold-wave | 5 | 140 | cold-wave1 |
| heat-wave | 5 | 165 | heat-wave1 |
| earthquake | 10 | 361 | ground movement, tsunami |
| flood | 17 | 522 | coastal flood, flash flood, general flood, riverine flood |
| mass movement (wet) | 6 | 165 | landslide wet, mudslide |
| storm | 55 | 1784 | 11 subtypes (hail, tornado, etc.) |
| volcanic activity | 15 | 409 | ash fall, general activity, lava flow, pyroclastic flow |
| wildfire | 14 | 441 | forest fire, general wildfire, land fire |
| **Total** | **127** | **3987** | |

### VQA Sample Fields

| Field | Description |
|-------|-------------|
| `Question_id` | Unique identifier |
| `Task` | Task category (Early Warning, Real-time Assessment, Retrospective Analysis) |
| `Subtask` | Subtask with timestep, e.g. "Risk Detection (t1)" |
| `Text` | Question text |
| `Image` | Comma-separated relative image paths |
| `Stations` | Weather station sensor data (JSON, optional) |
| `Ground truth` | Standard answer |

### Timestep Convention

- `t1` = 14 days before event start
- `t15` = event start day
- `t_number = (image_date - event_start_date).days + 15`

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies:
- `openai==2.21.0`
- `Pillow==12.1.1`

---

## Quick Start

**Configure API credentials:**

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
```

**Run evaluation using run.sh:**

```bash
# Edit DATA_ROOT in run.sh to point to your dataset path
bash run.sh
```

**Run evaluate.py directly:**

```bash
python evaluate.py \
  --dataset-root /path/to/ObsCrisis-Bench/storm_json \
  --raw-data-base-path /path/to/ObsCrisis-Bench \
  --target-file All.json \
  --model-name gpt-4o \
  --max-workers 64 \
  --image-max-num 500 \
  --total-max-size 47185920 \
  --temperature 0.1 \
  --max-tokens 300
```

**Resume interrupted evaluation:**

```bash
python evaluate.py <other args> \
  --resume \
  --resume-file output/eval_result_<timestamp>.json
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-root` | required | Directory containing the JSON evaluation file |
| `--raw-data-base-path` | required | Root directory of satellite imagery |
| `--target-file` | `All.json` | Evaluation JSON filename |
| `--model-name` | `gpt-4o` | Model name |
| `--max-workers` | `64` | Concurrent thread count |
| `--image-max-num` | `500` | Max images per request |
| `--total-max-size` | `47185920` (45 MB) | Max total image size per request (bytes) |
| `--temperature` | `0.1` | Model temperature |
| `--max-tokens` | `300` | Max output tokens |
| `--resume` | — | Enable resume mode |
| `--resume-file` | — | Previous result file to resume from |

---

## Supported Models

Pre-configured in `run.sh` with recommended parameters:

| Model | Max Images | Temperature | Notes |
|-------|-----------|-------------|-------|
| `gpt-4o` | 500 | 0.1 | OpenAI API |
| `gemini-3-pro-preview-thinking` | 600 | 1.0 | Supports `reasoning_content` |
| `qwen3.5-397b-a17b` | 250 | 0.6 | Alibaba Cloud API |

All models are accessed via OpenAI-compatible API. Extend as needed.

---

## Output Format

Results are saved to `output/eval_result_<dataset>_<model>_<file>_<timestamp>.json`:

```json
{
  "summary": {
    "overall": 0.72,
    "n": 3987,
    "by_task": {
      "Early Warning": {
        "overall": 0.72,
        "n": 165,
        "by_subtask": {
          "Risk Detection": {
            "overall": 0.85,
            "n": 45,
            "by_timestep": {
              "t1": {"score": 0.90, "n": 15},
              "t8": {"score": 0.83, "n": 15},
              "t12": {"score": 0.82, "n": 15}
            }
          }
        }
      }
    }
  },
  "details": [
    {
      "id": "Q001",
      "task": "Early Warning",
      "subtask": "Risk Detection (t1)",
      "ground_truth": "Yes",
      "prediction": "Yes",
      "score": 1.0,
      "score_type": "boolean",
      "raw_response": "...",
      "raw_reasoning_content": "..."
    }
  ]
}
```

### Scoring Rules

| Type | Condition | Rule |
|------|-----------|------|
| Boolean | Answer is yes/no | Exact match: 1.0 or 0.0 |
| Numeric | Answer parses to a number | Error ≤10% → 1.0, >20% → 0.0, linear decay between |
| Classification | Alphabetic string, length < 30 | Exact match → 1.0, contains match → 0.8 |
| Text | Other | Jaccard overlap on word sets |

---

## Utility Scripts

```bash
# Check output result file integrity
python scripts/check.py

# Analyze dataset statistics (task distribution, image counts, etc.)
python scripts/data_analyse.py
```

---

## License

This project is licensed under the MIT License.
