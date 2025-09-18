#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
code_summary.py
---------------
Quét toàn bộ file Python trong 1 thư mục (bao gồm subfolder),
ghép lại thành một file duy nhất để tiện mang đi gửi ChatGPT.

Usage:
    python code_summary.py --input ./my_project --output ./summary_all.py
"""

import argparse
from pathlib import Path

def collect_python_files(folder: Path):
    """Trả về danh sách tất cả file .py trong folder (bao gồm subfolder)."""
    return sorted(folder.rglob("*.py"))

def merge_files(files, output_file: Path):
    with output_file.open("w", encoding="utf-8") as fout:
        for f in files:
            rel_path = f.relative_to(output_file.parent.resolve())
            header = f"\n\n# ===== File: {rel_path} =====\n\n"
            fout.write(header)
            try:
                content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = f.read_text(encoding="latin-1")
            fout.write(content)
            fout.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Tổng hợp code Python thành 1 file duy nhất.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Thư mục chứa code Python")
    parser.add_argument("--output", "-o", type=str, default="code_summary_output.py",
                        help="File output (mặc định: code_summary_output.py)")
    args = parser.parse_args()

    input_folder = Path(args.input).resolve()
    output_file = Path(args.output).resolve()

    if not input_folder.is_dir():
        raise ValueError(f"Input folder không hợp lệ: {input_folder}")

    files = collect_python_files(input_folder)
    print(f"Found {len(files)} Python files in {input_folder}")

    merge_files(files, output_file)
    print(f"✅ Done. Code summary saved to {output_file}")

if __name__ == "__main__":
    main()
