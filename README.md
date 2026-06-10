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
- Disaster subtype statistics: **Disaster Type → Disaster Subtype → Task → Subtask → Timestep**

---

## Dataset

The benchmark dataset is available on HuggingFace: [ObsCrisis-Bench](https://huggingface.co/datasets/YYQ898/ObsCrisis-Bench)

### Dataset Statistics (After Cleaning)

| Category | Events | VQA Samples |
|----------|--------|-------------|
| cold-wave | 5 | 140 |
| earthquake | 10 | 417 |
| flood | 17 | 523 |
| heat-wave | 5 | 165 |
| mass movement (wet) | 6 | 189 |
| storm | 55 | 1,852 |
| volcanic activity | 15 | 450 |
| wildfire | 14 | 466 |
| **Total** | **127** | **4,202** |

**Note**: 397 error samples with incorrect tmax labels have been removed from the dataset.

### Task Distribution

| Task Category | Samples | Percentage |
|---------------|---------|------------|
| Early Warning | 1,424 | 33.89% |
| Impact Assessment | 2,565 | 61.04% |
| Recovery Assessment | 213 | 5.07% |

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

### VQA Sample Fields

| Field | Description |
|-------|-------------|
| `Question_id` | Unique identifier |
| `Task` | Task category (Early Warning, Impact Assessment, Recovery Assessment) |
| `Subtask` | Subtask with timestep, e.g. "Risk Detection (t1)" |
| `Text` | Question text |
| `Image` | Comma-separated relative image paths |
| `Stations` | Weather station sensor data (JSON, optional) |
| `Ground truth` | Standard answer |

### Timestep Convention

- `t1` = 14 days before event start
- `t8` = 7 days before event start
- `t12` = 3 days before event start
- `t15` = event start day
- `tmax` = maximum impact time
- `t_number = (image_date - event_start_date).days + 15`

---

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `openai==2.21.0`
- `Pillow==12.1.1`
- `huggingface-hub` (for uploading to HuggingFace)

### Download Dataset

```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download YYQ898/ObsCrisis-Bench --repo-type dataset --local-dir ./test

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="YYQ898/ObsCrisis-Bench", repo_type="dataset", local_dir="./test")
```

---

## Quick Start

### 1. Configure API Credentials

**Option A: Environment Variables (Recommended)**
```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key-here
```

**Option B: Edit run.sh**
```bash
# Edit run.sh and uncomment/set these lines:
export OPENAI_BASE_URL="YOUR_API_BASE_URL"
export OPENAI_API_KEY="YOUR_API_KEY"
```

### 2. Run Evaluation

**Using run.sh (Recommended)**
```bash
# Edit DATA_ROOT in run.sh to point to your dataset
bash run.sh
```

**Using evaluate.py directly**
```bash
python evaluate.py \
  --dataset-root ./test/storm_json \
  --raw-data-base-path ./test \
  --target-file All.json \
  --model-name gpt-4o \
  --max-workers 64 \
  --image-max-num 500 \
  --total-max-size 47185920 \
  --temperature 0.1 \
  --max-tokens 300
```

### 3. Resume Interrupted Evaluation

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
    "n": 4202,
    "total_samples": 4220,
    "api_errors": 18,
    "by_disaster_type": {
      "earthquake": {
        "overall": 0.41,
        "n": 397,
        "by_subtype": {
          "ground movement": {
            "overall": 0.35,
            "n": 217,
            "by_task": {
              "Early Warning": {
                "overall": 0.39,
                "n": 112,
                "by_subtask": {
                  "Risk Detection": {
                    "overall": 0.85,
                    "n": 38,
                    "by_timestep": {
                      "t1": {"score": 0.90, "n": 10},
                      "t8": {"score": 0.83, "n": 10},
                      "t12": {"score": 0.82, "n": 10}
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "by_task": {
      "Early Warning": {
        "overall": 0.72,
        "n": 1424,
        "by_subtask": {
          "Risk Detection": {
            "overall": 0.85,
            "n": 356,
            "by_timestep": {
              "t1": {"score": 0.90, "n": 89},
              "t8": {"score": 0.83, "n": 89},
              "t12": {"score": 0.82, "n": 89}
            }
          }
        }
      }
    }
  },
  "details": [
    {
      "id": "Early/Detection/t1/storm/2015-0069-TZA",
      "question": "...",
      "ground_truth": "Yes",
      "prediction": "Yes",
      "score": 1.0,
      "raw_response": "..."
    }
  ]
}
```

---

## Utility Scripts

### 1. Clean Error Samples

Remove samples with incorrect tmax labels from the dataset:

```bash
python scripts/clean_error_samples.py \
  --dataset-dir ./test \
  --dry-run  # Only identify, don't delete

# Actually delete
python scripts/clean_error_samples.py \
  --dataset-dir ./test \
  --eval-results output/eval_result_earthquake_final.json output/eval_result_cold-wave.json
```

**What it does**:
- Identifies samples with Question_id containing "tmax" but using incorrect time steps
- Removes these error samples from both dataset JSON and evaluation result JSON
- Generates a cleaning report

### 2. Recompute Statistics

Recompute statistics from evaluation results (e.g., after cleaning):

```bash
python scripts/recompute_statistics.py \
  --eval-file output/eval_result_earthquake_final.json \
  --output output/eval_result_earthquake_final_cleaned.json
```

**What it does**:
- Reads evaluation result JSON
- Filters out API Error samples
- Recomputes all statistics (overall, by task, by subtask, by timestep, by disaster type, by disaster subtype)
- Saves to new JSON file

### 3. Upload to HuggingFace

Upload the dataset to HuggingFace:

```bash
# Set HF_TOKEN environment variable
export HF_TOKEN=your_huggingface_token

# Upload
python scripts/upload_to_huggingface.py \
  --folder-path ./test \
  --repo-id YYQ898/ObsCrisis-Bench \
  --commit-message "Update dataset"
```

**Or provide token directly**:
```bash
python scripts/upload_to_huggingface.py \
  --token YOUR_HF_TOKEN \
  --folder-path ./test \
  --repo-id YYQ898/ObsCrisis-Bench
```

---

## Example Output Files

The `output/` directory contains example evaluation results:

- `eval_result_earthquake_final_cleaned.json` - Earthquake evaluation results (cleaned)
- `eval_result_cold-wave_json_gpt-5.5_All.json_20260610_144636_cleaned.json` - Cold-wave evaluation results (cleaned)

---

## Project Structure

```
Extreme-events/
├── evaluate.py              # Main evaluation script
├── run.sh                   # Batch evaluation script
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── README.md                # This file
├── .gitignore              # Git ignore rules
├── scripts/
│   ├── clean_error_samples.py      # Clean error samples
│   ├── recompute_statistics.py     # Recompute statistics
│   └── upload_to_huggingface.py    # Upload to HuggingFace
├── output/                  # Example output files
│   ├── eval_result_earthquake_final_cleaned.json
│   └── eval_result_cold-wave_json_gpt-5.5_All.json_20260610_144636_cleaned.json
└── test/                    # Dataset directory (not in Git)
    ├── README.md           # Dataset README
    ├── cold-wave_json/
    ├── earthquake_json/
    ├── flood_json/
    ├── heat-wave_json/
    ├── mass movement (wet)_json/
    ├── storm/
    ├── storm_json/
    ├── volcanic activity_json/
    └── wildfire_json/
```

---

## Common Issues

### 1. API Error Samples

Some samples may fail due to API errors. These are automatically filtered out during statistics computation:
- Check `summary.api_errors` in the output JSON
- Check `summary.total_samples` vs `summary.n` to see how many were filtered

### 2. Resume Interrupted Evaluation

If evaluation is interrupted:
```bash
python evaluate.py <args> --resume --resume-file output/eval_result_<timestamp>.json
```

### 3. Large Dataset Upload

For large datasets, use `upload_large_folder` (automatically used by the script):
```bash
python scripts/upload_to_huggingface.py --folder-path ./test
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{obscrisis-bench,
  title={ObsCrisis-Bench: A Multimodal Benchmark for Extreme Weather Event Analysis},
  author={YYQ898},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/YYQ898/ObsCrisis-Bench}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/YYQ898/ObsCrisis-Bench).
