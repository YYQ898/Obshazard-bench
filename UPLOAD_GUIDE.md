# 上传指南

## 一、上传数据集到 HuggingFace

### 1. 安装 huggingface-hub
```bash
pip install huggingface-hub
```

### 2. 登录 HuggingFace
```bash
huggingface-cli login
# 或使用 token
huggingface-cli login --token YOUR_HF_TOKEN
```

### 3. 上传数据集
数据集目录：`d:\ObsCrisis-Bench\Extreme-events\test`
目标仓库：`https://huggingface.co/datasets/YYQ898/ObsCrisis-Bench`

#### 方法 A：使用 huggingface-cli（推荐）
```bash
# 从 Extreme-events 目录运行
cd d:\ObsCrisis-Bench\Extreme-events

# 上传整个 test 目录到 HuggingFace
huggingface-cli upload YYQ898/ObsCrisis-Bench test . --repo-type dataset
```

#### 方法 B：使用 Python 脚本
```python
from huggingface_hub import HfApi

api = HfApi()

# 上传整个文件夹
api.upload_folder(
    folder_path="d:/ObsCrisis-Bench/Extreme-events/test",
    repo_id="YYQ898/ObsCrisis-Bench",
    repo_type="dataset",
    commit_message="Upload ObsCrisis-Bench dataset"
)
```

---

## 二、上传代码到 GitHub

### 1. 初始化 Git 仓库（如果还没有）
```bash
cd d:\ObsCrisis-Bench\Extreme-events

# 初始化 git（如果需要）
git init

# 添加远程仓库
git remote add origin https://github.com/YYQ898/ObsCrisis-Bench.git
```

### 2. 更新 .gitignore
确保 `.gitignore` 包含以下内容，以排除数据集：
```
# 数据集目录
test/

# 输出目录
output/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# 系统文件
.DS_Store
.env
*.bak
*.bak2
_*

# IDE
.vscode/
.idea/
```

### 3. 提交并推送代码
```bash
# 添加所有文件（test 目录会被 .gitignore 排除）
git add .

# 提交
git commit -m "Clean up sensitive information and prepare for release"

# 推送到 GitHub
git push -u origin main
# 或如果远程已有内容
git push origin main
```

---

## 三、验证上传

### HuggingFace 数据集
访问：https://huggingface.co/datasets/YYQ898/ObsCrisis-Bench
确认数据集文件都已上传

### GitHub 代码
访问：https://github.com/YYQ898/ObsCrisis-Bench
确认代码文件都已上传，且不包含 test 数据集目录

---

## 四、注意事项

1. **敏感信息已清理**：
   - ✅ `run.sh` - API URL 和 Key 已移除
   - ✅ `data_analyse.py` - 硬编码路径已改为参数
   - ✅ `rlaunch_cpu.sh` - 敏感信息已移除
   - ✅ `evaluate.py` - 使用环境变量，无硬编码敏感信息

2. **数据集结构**：
   - 数据集位于 `test/` 目录
   - 包含多个灾害类型的 JSON 文件和图像数据
   - storm 目录包含 hail 子类型的图像数据

3. **代码结构**（不包含数据集）：
   - `evaluate.py` - 主评测脚本
   - `scripts/` - 辅助脚本
   - `requirements.txt` - 依赖
   - `README.md` - 说明文档
   - `.gitignore` - Git 忽略配置
