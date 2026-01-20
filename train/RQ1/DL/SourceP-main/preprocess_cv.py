import json
import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import csv

# --- 1. 你的原始文件 ---
SOURCE_FILE_PATH = './dataset/dataset.jsonl'
LABEL_FILE_PATH = './dataset/label_6188.csv'

# --- 2. 配置 ---
OUTPUT_DIR = './dataset'
MASTER_DATA_JSONL = os.path.join(OUTPUT_DIR, 'dataset.jsonl')

TEST_SPLIT_RATIO = 0.20
VALIDATION_SPLIT_RATIO = 0.10
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_simple_split():
    print("开始读取原始数据...")
    all_data = []
    labels = []

    try:
        with open(SOURCE_FILE_PATH, 'r', encoding='utf-8') as f_source, \
                open(LABEL_FILE_PATH, 'r', encoding='utf-8') as f_label_file:

            label_reader = csv.reader(f_label_file)

            try:
                next(label_reader)  # 跳过表头 'address,label'
            except StopIteration:
                print("错误: 标签文件为空。")
                return

            for i, (source_line, label_row) in enumerate(zip(f_source, label_reader)):
                unique_id = f"line_{i}"

                try:
                    label = int(label_row[1].strip())
                except (ValueError, IndexError):
                    print(f"警告: 第 {i + 1} 行标签值 '{label_row}' 格式错误，已跳过。")
                    continue

                try:
                    source_data = json.loads(source_line)
                    all_data.append({
                        'idx': unique_id,
                        'sourcecode': source_data['sourcecode'],
                        'label': label
                    })
                    labels.append(label)
                except json.JSONDecodeError:
                    print(f"警告: 第 {i + 1} 行 {SOURCE_FILE_PATH} JSON 格式错误，已跳过。")
                except KeyError:
                    print(f"警告: 第 {i + 1} 行 {SOURCE_FILE_PATH} 缺少 'sourcecode' 键，已跳过。")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}")
        return

    print(f"总共读取 {len(all_data)} 条数据。")

    # --- 3. 创建主 data.jsonl ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"正在创建主 data.jsonl 到: {MASTER_DATA_JSONL}")
    with open(MASTER_DATA_JSONL, 'w', encoding='utf-8') as f_out:
        for item in all_data:
            f_out.write(json.dumps({'idx': item['idx'], 'sourcecode': item['sourcecode']}) + '\n')

    # --- 4. 执行两次 Stratified Split (分层拆分) ---
    # 第一次拆分: 80% (Train+Valid) 和 20% (Test)
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        all_data, labels,
        test_size=TEST_SPLIT_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # 第二次拆分: 从 80% 中拆分 70% (Train) 和 10% (Valid)
    # 10% / 80% = 0.125
    val_ratio_in_train_val = VALIDATION_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_train_val, labels_train_val,
        test_size=val_ratio_in_train_val,
        stratify=labels_train_val,
        random_state=RANDOM_SEED
    )

    # --- 5. 写入文件 ---
    write_label_file(data_train, os.path.join(OUTPUT_DIR, 'train.txt'))
    write_label_file(data_val, os.path.join(OUTPUT_DIR, 'valid.txt'))
    write_label_file(data_test, os.path.join(OUTPUT_DIR, 'test.txt'))

    print("=" * 30)
    print("所有数据已生成完毕！")
    print(f"数据存放在: {OUTPUT_DIR}")
    print(f"拆分统计:")
    print(f"  训练集 (train.txt): {len(data_train)} 条")
    print(f"  验证集 (valid.txt): {len(data_val)} 条")
    print(f"  测试集 (test.txt):  {len(data_test)} 条")


def write_label_file(data_items, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_items:
            f.write(f"{item['idx']}\t{item['label']}\n")


if __name__ == "__main__":
    create_simple_split()