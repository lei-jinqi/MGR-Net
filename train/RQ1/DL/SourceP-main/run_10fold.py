import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import subprocess
import shutil
from evaluator import read_answers, read_predictions, calculate_scores

# 配置路径
DATA_DIR = "dataset"
RESULT_DIR = "result"
ALL_DATA_FILE = os.path.join(DATA_DIR, "all_data.txt")  # 你需要把所有数据合并到这个文件
CONFIG_NAME = "parameters"  # 对应 parameters.cfg


# 如果没有 all_data.txt，你可以取消下面注释，从现有的 train/valid/test 合并
def merge_all_data():
    data = []
    for fname in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data.extend(f.readlines())
    # 去重并保存
    data = list(set(data))
    with open(ALL_DATA_FILE, 'w') as f:
        f.writelines(data)
    print(f"Merged {len(data)} samples into {ALL_DATA_FILE}")


# 如果尚未合并数据，执行一次合并
if not os.path.exists(ALL_DATA_FILE):
    merge_all_data()


def run_10_fold():
    # 1. 读取所有数据
    print(f"Loading data from {ALL_DATA_FILE}...")
    with open(ALL_DATA_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    X = [line.split('\t')[0] for line in lines]  # ID
    y = [int(line.split('\t')[1]) for line in lines]  # Label

    X = np.array(X)
    y = np.array(y)
    lines = np.array(lines)

    # 2. 准备十折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_results = []

    # 清理旧的结果
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'=' * 20} Running Fold {fold + 1}/10 {'=' * 20}")

        # 再次切分出验证集 (例如从训练集中划出 10% 做验证)
        # 注意：run_classifier 需要 valid set 来保存最佳模型
        inner_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        train_idx, val_idx = next(inner_skf.split(X[train_val_idx], y[train_val_idx]))

        # 映射回原始索引
        final_train_idx = train_val_idx[train_idx]
        final_val_idx = train_val_idx[val_idx]

        # 3. 写入当前折的数据文件
        # run_classifier 读取的是 dataset/train.txt 等固定路径
        np.savetxt(os.path.join(DATA_DIR, "train.txt"), lines[final_train_idx], fmt='%s')
        np.savetxt(os.path.join(DATA_DIR, "valid.txt"), lines[final_val_idx], fmt='%s')
        np.savetxt(os.path.join(DATA_DIR, "test.txt"), lines[test_idx], fmt='%s')

        print(f"Train size: {len(final_train_idx)}, Valid size: {len(final_val_idx)}, Test size: {len(test_idx)}")

        # 4. 调用基线训练命令
        # 相当于命令行执行: python run_classifier.py parameters
        cmd = [sys.executable, "run_classifier.py", CONFIG_NAME]
        subprocess.check_call(cmd)

        # 5. 评估当前折结果
        # predictions.txt 是 run_classifier.py 生成的
        pred_file = os.path.join(RESULT_DIR, "predictions.txt")
        test_file = os.path.join(DATA_DIR, "test.txt")

        # 备份当前折的预测结果（可选）
        shutil.copy(pred_file, os.path.join(RESULT_DIR, f"predictions_fold_{fold + 1}.txt"))

        # 使用 evaluator 计算指标
        answers = read_answers(test_file)
        predictions = read_predictions([pred_file])[0]  # read_predictions returns a list

        # calculate_scores 返回 list of dicts
        # 注意 evaluator.py 中的 calculate_scores 逻辑需要 predictions 是 list of dicts
        # 这里我们手动适配一下 evaluator 的逻辑
        score = calculate_scores(answers, [predictions])[0]

        print(f"Fold {fold + 1} Result: {score}")
        fold_results.append(score)

    # 6. 输出平均结果
    print("\n" + "=" * 20 + " 10-Fold CV Summary " + "=" * 20)
    metrics = fold_results[0].keys()
    final_avg = {}

    # 打印每折详情
    df = pd.DataFrame(fold_results)
    print(df)

    print("-" * 40)
    for m in metrics:
        avg_val = np.mean([res[m] for res in fold_results])
        std_val = np.std([res[m] for res in fold_results])
        final_avg[m] = avg_val
        print(f"{m:<15}: {avg_val:.4f} (+/- {std_val:.4f})")


if __name__ == "__main__":
    run_10_fold()