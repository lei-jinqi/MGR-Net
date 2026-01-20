import os

file_path = "dataset/all_data.txt"
output_path = "dataset/all_data_clean.txt"

print(f"正在检查文件: {file_path} ...")

valid_lines = []
dirty_count = 0
labels_found = {}

if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}。请确保你已经运行过一次 run_10fold.py 来生成它。")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue  # 跳过空行

        parts = line.split('\t')

        # 检查 1: 必须严格只有 2 列
        if len(parts) != 2:
            print(f"[行 {line_num}] 格式错误 (列数={len(parts)}): {line}")
            dirty_count += 1
            continue

        url, label = parts

        # 检查 2: 标签必须是 0 或 1
        if label not in ['0', '1']:
            print(f"[行 {line_num}] 标签异常: {line}")
            dirty_count += 1
            continue

        # 统计标签分布
        labels_found[label] = labels_found.get(label, 0) + 1
        valid_lines.append(line)

# 保存清洗后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(valid_lines) + '\n')

print("-" * 30)
print(f"清洗完成！")
print(f"共发现并删除了 {dirty_count} 行脏数据。")
print(f"剩余有效数据: {len(valid_lines)} 行。")
print(f"标签分布: {labels_found}")
print(f"文件已覆盖保存至: {file_path}")
print("-" * 30)