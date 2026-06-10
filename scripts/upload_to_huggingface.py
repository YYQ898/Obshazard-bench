#!/usr/bin/env python3
"""
上传数据集到 HuggingFace 的脚本
用法：python upload_to_huggingface.py --token YOUR_HF_TOKEN
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument("--token", type=str, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--repo-id", type=str, default="YYQ898/ObsCrisis-Bench", help="Target repo ID")
    parser.add_argument("--folder-path", type=str, default=None, help="Folder to upload (default: ./test)")
    parser.add_argument("--commit-message", type=str, default="Upload ObsCrisis-Bench dataset", help="Commit message")
    args = parser.parse_args()

    # 确定上传路径
    if args.folder_path is None:
        script_dir = Path(__file__).parent
        folder_path = script_dir / "test"
    else:
        folder_path = Path(args.folder_path)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return

    # 初始化 API
    api = HfApi(token=args.token)

    print(f"Uploading folder: {folder_path}")
    print(f"Target repo: {args.repo_id}")
    print(f"Commit message: {args.commit_message}")
    print("Starting upload...")

    try:
        # 使用 upload_large_folder 处理大文件夹
        # 注意：upload_large_folder 不支持 commit_message 参数
        api.upload_large_folder(
            folder_path=str(folder_path),
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        print("\n[OK] Upload completed successfully!")
        print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
