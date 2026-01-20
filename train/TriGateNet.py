import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, \
    average_precision_score
from tqdm import tqdm
import ast
import json
import warnings
from collections import Counter, defaultdict
import os
import datetime
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# --- 新增：日志功能 ---
RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(RESULT_DIR, f"trigatenet_10fold_{current_time}.log")


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(str(msg) + "\n")


class Config:
    OPCODE_FILE = 'dataset/lists/filteredcfglists.csv'
    CFG_FILE = 'dataset/blocks/filteredcfgs.csv'
    DFG_FILE = 'dataset/dfg.csv'
    LABEL_FILE = 'dataset/label_6188.csv'

    EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 128
    GNN_HIDDEN_DIM = 128
    FUSION_DIM = 128

    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    EARLY_STOPPING_PATIENCE = 7
    WEIGHT_DECAY = 2e-2
    RANDOM_SEED = 42

    OPCODE_MIN_FREQ = 1
    DFG_MIN_FREQ = 4

    N_SPLITS = 10


def flatten_corpus_iterative(nested_list):
    if not isinstance(nested_list, list):
        return [nested_list]
    flattened = []
    stack = list(reversed(nested_list))
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            flattened.append(item)
    return flattened


class Tokenizer:
    def __init__(self, pad_token='<pad>', unk_token='<unk>', min_freq=1):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.min_freq = min_freq
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_built = False

    def build_vocab(self, flat_corpus):
        if self.vocab_built: return
        token_counts = Counter(flat_corpus)
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        for token, count in token_counts.items():
            if count >= self.min_freq:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_built = True
        log(f"词汇表构建完成。总Token数量 (>= {self.min_freq} 词频): {self.get_vocab_size()}")

    def tokens_to_ids(self, tokens):
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def get_vocab_size(self):
        return len(self.token_to_id)

    def get_pad_id(self):
        return self.token_to_id[self.pad_token]


class PonziDataset(Dataset):
    def __init__(self, dataframe, op_tokenizer, dfg_tokenizer):
        self.df = dataframe
        self.op_tokenizer = op_tokenizer
        self.dfg_tokenizer = dfg_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            op_sequence_nested = ast.literal_eval(row['oplist'])
            op_sequence = flatten_corpus_iterative(op_sequence_nested)
        except (ValueError, SyntaxError):
            op_sequence = []
        op_ids = self.op_tokenizer.tokens_to_ids(op_sequence)
        op_tensor = torch.tensor(op_ids, dtype=torch.long)

        try:
            full_cfg_json = json.loads(row['basicblocks'])
            cfg_graph = self._create_cfg_graph(full_cfg_json)
        except (json.JSONDecodeError, TypeError, KeyError):
            cfg_graph = self._create_empty_graph(is_cfg=True)

        try:
            dfg_nodes_nested = json.loads(row['node_feature'])
            dfg_nodes = flatten_corpus_iterative(dfg_nodes_nested)
            dfg_edges = json.loads(row['edge_index'])
            dfg_graph = self._create_dfg_graph(dfg_nodes, dfg_edges)
        except (json.JSONDecodeError, TypeError, KeyError):
            dfg_graph = self._create_empty_graph(is_cfg=False)

        label = torch.tensor(row['label'], dtype=torch.float)
        return op_tensor, cfg_graph, dfg_graph, label

    def _create_cfg_graph(self, full_cfg_json):
        if 'basicblocks' not in full_cfg_json or not full_cfg_json['basicblocks']:
            return self._create_empty_graph(is_cfg=True)
        actual_blocks = full_cfg_json['basicblocks']
        if not isinstance(actual_blocks, dict):
            return self._create_empty_graph(is_cfg=True)

        block_ids = sorted(actual_blocks.keys(), key=int)
        block_id_to_idx = {block_id: i for i, block_id in enumerate(block_ids)}
        node_features = []
        edge_list = []

        for block_id in block_ids:
            block = actual_blocks[block_id]
            opcodes = [inst['opname'] for inst in block.get('instructions', [])]
            node_features.append(torch.tensor(self.op_tokenizer.tokens_to_ids(opcodes), dtype=torch.long))
            current_idx = block_id_to_idx[block_id]
            for succ_id in block.get('successors', []):
                if str(succ_id) in block_id_to_idx:
                    succ_idx = block_id_to_idx[str(succ_id)]
                    edge_list.append([current_idx, succ_idx])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty(2, 0,
                                                                                                              dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index)

    def _create_dfg_graph(self, nodes, edges):
        if not nodes:
            return self._create_empty_graph(is_cfg=False)
        node_ids = self.dfg_tokenizer.tokens_to_ids(nodes)
        x = torch.tensor(node_ids, dtype=torch.long).unsqueeze(1)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0,
                                                                                                      dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def _create_empty_graph(self, is_cfg):
        x_data = [] if is_cfg else torch.empty(0, 1, dtype=torch.long)
        return Data(x=x_data, edge_index=torch.empty(2, 0, dtype=torch.long))


def collate_fn(batch):
    op_tensors, cfg_graphs, dfg_graphs, labels = zip(*batch)
    op_padded = pad_sequence(op_tensors, batch_first=True, padding_value=0)
    cfg_batch = Batch.from_data_list(cfg_graphs)
    dfg_batch = Batch.from_data_list(dfg_graphs)
    labels_tensor = torch.stack(labels)
    return op_padded, cfg_batch, dfg_batch, labels_tensor


class OpcodeLSTM(nn.Module):
    def __init__(self, embedding_layer, hidden_dim):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = self.embedding.embedding_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_cat)


class BlockEncoder(nn.Module):
    def __init__(self, embedding_layer, hidden_dim):
        super().__init__()
        self.embedding = embedding_layer
        self.gru = nn.GRU(embedding_layer.embedding_dim, hidden_dim, batch_first=True)

    def forward(self, op_ids_tensor):
        embedded = self.embedding(op_ids_tensor.unsqueeze(0))
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0).squeeze(0)


class GraphGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.size(0) == 0:
            return torch.zeros(data.num_graphs, self.out_channels, device=edge_index.device)
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class GatedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim * 3, 3)
        self.output_linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, op_vec, cfg_vec, dfg_vec):
        combined = torch.cat([op_vec, cfg_vec, dfg_vec], dim=1)
        gates = torch.softmax(self.gate_linear(combined), dim=1)
        g_op, g_cfg, g_dfg = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1), gates[:, 2].unsqueeze(1)
        fused_vec = g_op * op_vec + g_cfg * cfg_vec + g_dfg * dfg_vec
        return self.tanh(self.output_linear(fused_vec))


class PonziDetector(nn.Module):
    def __init__(self, op_vocab_size, dfg_vocab_size, config):
        super().__init__()
        self.config = config
        self.opcode_embedding = nn.Embedding(op_vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        self.variable_embedding = nn.Embedding(dfg_vocab_size, config.EMBEDDING_DIM, padding_idx=0)
        self.cfg_block_encoder = BlockEncoder(self.opcode_embedding, config.GNN_HIDDEN_DIM)
        self.opcode_encoder = OpcodeLSTM(self.opcode_embedding, config.LSTM_HIDDEN_DIM)
        self.cfg_gnn = GraphGNN(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM)
        self.dfg_gnn = GraphGNN(config.EMBEDDING_DIM, config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM)
        self.dropout = nn.Dropout(0.5)
        self.fusion = GatedFusion(config.LSTM_HIDDEN_DIM, config.FUSION_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.FUSION_DIM // 2, 1)
        )

    def forward(self, op_data, cfg_data, dfg_data):
        op_vec = self.opcode_encoder(op_data)

        cfg_encoded_nodes = []
        if hasattr(cfg_data, 'x') and isinstance(cfg_data.x, list):
            for node_op_ids in cfg_data.x:
                if isinstance(node_op_ids, torch.Tensor) and node_op_ids.numel() > 0:
                    encoded_node = self.cfg_block_encoder(node_op_ids.to(op_data.device))
                    cfg_encoded_nodes.append(encoded_node)
        if cfg_encoded_nodes:
            cfg_data.x = torch.stack(cfg_encoded_nodes)
        else:
            cfg_data.x = torch.empty(0, self.config.GNN_HIDDEN_DIM, device=op_data.device)
        cfg_vec = self.cfg_gnn(cfg_data)

        if dfg_data.x.numel() > 0:
            dfg_data.x = self.variable_embedding(dfg_data.x.squeeze(-1))
        else:
            dfg_data.x = torch.empty(0, self.config.EMBEDDING_DIM, device=op_data.device)
        dfg_vec = self.dfg_gnn(dfg_data)

        op_vec_d = self.dropout(op_vec)
        cfg_vec_d = self.dropout(cfg_vec)
        dfg_vec_d = self.dropout(dfg_vec)
        fused_vec = self.fusion(op_vec_d, cfg_vec_d, dfg_vec_d)
        output = self.classifier(fused_vec)
        return torch.sigmoid(output.squeeze(1))


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for op_data, cfg_data, dfg_data, labels in dataloader:
        op_data, labels = op_data.to(device), labels.to(device)
        cfg_data, dfg_data = cfg_data.to(device), dfg_data.to(device)
        optimizer.zero_grad()
        outputs = model(op_data, cfg_data, dfg_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for op_data, cfg_data, dfg_data, labels in dataloader:
            op_data, labels = op_data.to(device), labels.to(device)
            cfg_data, dfg_data = cfg_data.to(device), dfg_data.to(device)
            outputs = model(op_data, cfg_data, dfg_data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    auc_pr = average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

    best_f1, best_threshold = 0, 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        preds_binary = (all_preds > threshold).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    preds_at_best_f1 = (all_preds > best_threshold).astype(int)
    precision = precision_score(all_labels, preds_at_best_f1, zero_division=0)
    recall = recall_score(all_labels, preds_at_best_f1, zero_division=0)
    accuracy = accuracy_score(all_labels, preds_at_best_f1)

    return total_loss / len(dataloader), auc, auc_pr, best_f1, precision, recall, accuracy, best_threshold


def main():
    config = Config()
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"日志文件路径: {LOG_FILE}")
    log(f"使用设备: {device}")

    log("--- 步骤 1: 加载和合并数据 ---")
    try:

        df_op = pd.read_csv(config.OPCODE_FILE).rename(columns={'opcodes': 'oplist'})
        df_cfg = pd.read_csv(config.CFG_FILE).rename(columns={'nodes': 'basicblocks'})
        df_dfg = pd.read_csv(config.DFG_FILE)
        df_label = pd.read_csv(config.LABEL_FILE)
    except FileNotFoundError as e:
        log(f"错误: {e}. 请确保所有CSV文件都在正确的路径下。")
        return

    df = pd.merge(df_op, df_cfg, on='address', how='outer')
    df = pd.merge(df, df_dfg, on='address', how='outer')
    df = pd.merge(df, df_label, on='address', how='outer')

    required_columns = ['oplist', 'basicblocks', 'node_feature', 'edge_index', 'label']
    df.dropna(subset=required_columns, inplace=True)
    df = df.reset_index(drop=True)
    log(f"数据加载与清洗完成。最终有效合约数量: {len(df)}")

    log(f"\n--- 步骤 2: 初始化 {config.N_SPLITS}-折交叉验证 ---")
    y = df['label']
    kfold = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    all_fold_results = defaultdict(list)

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(df, y)):
        log(f"\n{'=' * 25} 折 {fold + 1}/{config.N_SPLITS} {'=' * 25}")

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        val_split_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_split_ratio,
            random_state=config.RANDOM_SEED,
            stratify=train_val_df['label']
        )
        log(f"当前折数据划分: {len(train_df)} 训练 / {len(val_df)} 验证 / {len(test_df)} 测试")

        op_corpus_nested = [ast.literal_eval(seq) for seq in train_df['oplist']]
        cfg_block_corpus_nested = [json.loads(nodes) for nodes in train_df['basicblocks']]
        cfg_opcodes = [inst['opname'] for block in cfg_block_corpus_nested if
                       'basicblocks' in block and isinstance(block['basicblocks'], dict) for b_id, b_content in
                       block['basicblocks'].items() for inst in b_content.get('instructions', [])]
        op_flat_corpus = flatten_corpus_iterative(op_corpus_nested) + cfg_opcodes

        dfg_node_corpus_nested = [json.loads(nodes) for nodes in train_df['node_feature']]
        dfg_flat_corpus = flatten_corpus_iterative(dfg_node_corpus_nested)

        op_tokenizer = Tokenizer(min_freq=config.OPCODE_MIN_FREQ)
        op_tokenizer.build_vocab(op_flat_corpus)
        dfg_tokenizer = Tokenizer(min_freq=config.DFG_MIN_FREQ)
        dfg_tokenizer.build_vocab(dfg_flat_corpus)

        train_dataset = PonziDataset(train_df, op_tokenizer, dfg_tokenizer)
        val_dataset = PonziDataset(val_df, op_tokenizer, dfg_tokenizer)
        test_dataset = PonziDataset(test_df, op_tokenizer, dfg_tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = PonziDetector(
            op_vocab_size=op_tokenizer.get_vocab_size(),
            dfg_vocab_size=dfg_tokenizer.get_vocab_size(),
            config=config
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=False)

        best_val_f1 = 0
        patience_counter = 0
        best_model_path = os.path.join(RESULT_DIR, f'best_model_fold_{fold + 1}.pth')

        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auc, val_auc_pr, val_f1, val_p, val_r, val_acc, val_thresh = evaluate(model, val_loader,
                                                                                                criterion,
                                                                                                device)

            if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
                print(f"  -- 折 {fold + 1}, 轮 {epoch + 1}: Val F1: {val_f1:.4f}")

            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"  -- 折 {fold + 1}: EarlyStopping触发")
                    break

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_auc, test_auc_pr, test_f1, test_p, test_r, test_acc, test_thresh = evaluate(model, test_loader,
                                                                                                    criterion,
                                                                                                    device)

        log(f"  [折 {fold + 1} 结果] AUC: {test_auc:.4f} | F1: {test_f1:.4f} | P: {test_p:.4f} | R: {test_r:.4f}")

        all_fold_results['AUC'].append(test_auc)
        all_fold_results['AUC-PR'].append(test_auc_pr)
        all_fold_results['F1'].append(test_f1)
        all_fold_results['Precision'].append(test_p)
        all_fold_results['Recall'].append(test_r)
        all_fold_results['Accuracy'].append(test_acc)

        if os.path.exists(best_model_path):
            os.remove(best_model_path)

    log("\n\n" + "=" * 40)
    log(f"--- {config.N_SPLITS}-折交叉验证")
    log("=" * 40)

    log("以下列表用于 T-test / Wilcoxon Test:")
    for metric, values in all_fold_results.items():
        formatted_list = "[" + ", ".join([f"{v:.4f}" for v in values]) + "]"
        log(f"{metric}_list = {formatted_list}")

    log("\n平均指标统计:")
    metrics_df = pd.DataFrame(all_fold_results)

    summary_df = pd.DataFrame({
        'Mean': metrics_df.mean(),
        'Std': metrics_df.std()
    })
    log(summary_df.to_string())
    log(f"\n所有结果已保存至: {LOG_FILE}")


if __name__ == '__main__':
    main()
