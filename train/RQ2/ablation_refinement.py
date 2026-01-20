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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import ast
import json
import warnings
from collections import Counter
import os

warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    OPCODE_FILE = '../../dataset/lists/filteredcfglists.csv'
    CFG_FILE = '../../dataset/blocks/filteredcfgs.csv'
    DFG_FILE = '../../dataset/dfg.csv'
    LABEL_FILE = '../../dataset/label_6188.csv'

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
    N_SPLITS = 5

    OPCODE_MIN_FREQ = 1
    DFG_MIN_FREQ = 4


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
        print(f"Vocabulary built (>= {self.min_freq} freq): {self.get_vocab_size()} tokens")

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

    return total_loss / len(dataloader), auc, best_f1, precision, recall, accuracy, best_threshold


def main():
    base_config = Config()
    np.random.seed(base_config.RANDOM_SEED)
    torch.manual_seed(base_config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ablation_modes = ['no_cfg_pruning','no_dfg_pruning']
    final_results = {}

    for mode in ablation_modes:
        print(f"\n\n{'=' * 50}")
        print(f"--- Starting {base_config.N_SPLITS}-Fold CV for Refinement Ablation: {mode} ---")
        print(f"{'=' * 50}")

        fold_config = Config()
        if mode == 'no_cfg_pruning':
            fold_config.OPCODE_FILE = '../../dataset/lists/rawcfglists.csv'
            fold_config.CFG_FILE = '../../dataset/blocks/rawcfgs.csv'
            print("Running with: No CFG Pruning (using raw files)")
        elif mode == 'no_dfg_pruning':
            fold_config.DFG_MIN_FREQ = 1
            print("Running with: No DFG Pruning (DFG Min Freq = 1)")

        print("--- Step 1: Loading and merging data ---")
        try:
            df_op = pd.read_csv(fold_config.OPCODE_FILE).rename(columns={'opcodes': 'oplist'})
            df_cfg = pd.read_csv(fold_config.CFG_FILE).rename(columns={'nodes': 'basicblocks'})
            df_dfg = pd.read_csv(fold_config.DFG_FILE)
            df_label = pd.read_csv(fold_config.LABEL_FILE)
        except FileNotFoundError as e:
            print(f"Error: {e}. 停止. 请检查文件路径 {fold_config.OPCODE_FILE} 或 {fold_config.CFG_FILE}.")
            continue

        df = pd.merge(df_op, df_cfg, on='address', how='outer')
        df = pd.merge(df, df_dfg, on='address', how='outer')
        df = pd.merge(df, df_label, on='address', how='outer')

        required_columns = ['oplist', 'basicblocks', 'node_feature', 'edge_index', 'label']
        df.dropna(subset=required_columns, inplace=True)
        df = df.reset_index(drop=True)
        print(f"Data loaded and cleaned. Final effective contracts: {len(df)}")

        X = df.drop(columns=['label'])
        y = df['label']
        kfold = StratifiedKFold(n_splits=fold_config.N_SPLITS, shuffle=True, random_state=fold_config.RANDOM_SEED)

        fold_metrics = []

        for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X, y)):
            print(f"\n--- {mode} | Fold {fold + 1}/{fold_config.N_SPLITS} ---")

            train_val_df_fold = df.iloc[train_val_idx]
            test_df_fold = df.iloc[test_idx]

            train_df_fold, val_df_fold = train_test_split(
                train_val_df_fold,
                test_size=0.125,
                random_state=fold_config.RANDOM_SEED,
                stratify=train_val_df_fold['label']
            )

            print(
                f"Fold {fold + 1} split: {len(train_df_fold)} Train / {len(val_df_fold)} Val / {len(test_df_fold)} Test")

            op_corpus_nested = [ast.literal_eval(seq) for seq in train_df_fold['oplist']]
            cfg_block_corpus_nested = [json.loads(nodes) for nodes in train_df_fold['basicblocks']]
            cfg_opcodes = [inst['opname'] for block in cfg_block_corpus_nested if
                           'basicblocks' in block and isinstance(block['basicblocks'], dict) for b_id, b_content in
                           block['basicblocks'].items() for inst in b_content.get('instructions', [])]
            op_flat_corpus = flatten_corpus_iterative(op_corpus_nested) + cfg_opcodes
            dfg_node_corpus_nested = [json.loads(nodes) for nodes in train_df_fold['node_feature']]
            dfg_flat_corpus = flatten_corpus_iterative(dfg_node_corpus_nested)

            op_tokenizer = Tokenizer(min_freq=fold_config.OPCODE_MIN_FREQ)
            op_tokenizer.build_vocab(op_flat_corpus)

            dfg_tokenizer = Tokenizer(min_freq=fold_config.DFG_MIN_FREQ)
            dfg_tokenizer.build_vocab(dfg_flat_corpus)

            train_dataset = PonziDataset(train_df_fold, op_tokenizer, dfg_tokenizer)
            val_dataset = PonziDataset(val_df_fold, op_tokenizer, dfg_tokenizer)
            test_dataset = PonziDataset(test_df_fold, op_tokenizer, dfg_tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=fold_config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=fold_config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=fold_config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

            model = PonziDetector(
                op_vocab_size=op_tokenizer.get_vocab_size(),
                dfg_vocab_size=dfg_tokenizer.get_vocab_size(),
                config=fold_config
            ).to(device)

            criterion = nn.BCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=fold_config.LEARNING_RATE, weight_decay=fold_config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                             verbose=False)

            best_val_f1 = 0
            patience_counter = 0
            best_model_path = f'best_model_refinement_{mode}_fold_{fold}.pth'

            for epoch in range(fold_config.EPOCHS):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_auc, val_f1, val_p, val_r, val_acc, val_thresh = evaluate(model, val_loader, criterion,
                                                                                        device)

                if epoch % 5 == 0 or epoch == fold_config.EPOCHS - 1:
                    print(
                        f"  Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

                scheduler.step(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= fold_config.EARLY_STOPPING_PATIENCE:
                        print(f"  EarlyStopping triggered at epoch {epoch + 1}.")
                        break

            print(f"\n--- {mode} | Fold {fold + 1}/5 | Final Evaluation on Test Set ---")
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_auc, test_f1, test_p, test_r, test_acc, test_thresh = evaluate(model, test_loader,
                                                                                           criterion, device)

            fold_metrics.append({
                'AUC': test_auc,
                'F1': test_f1,
                'Precision': test_p,
                'Recall': test_r,
                'Accuracy': test_acc
            })
            print(f"  Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
            os.remove(best_model_path)

        df_fold_metrics = pd.DataFrame(fold_metrics)
        avg_metrics = df_fold_metrics.mean().to_dict()
        std_metrics = df_fold_metrics.std().to_dict()
        final_results[mode] = {'avg': avg_metrics, 'std': std_metrics}

        print(f"\n--- {mode} | Averaged Test Results Across 5 Folds ---")
        for metric in avg_metrics.keys():
            avg_val = avg_metrics[metric]
            std_val = std_metrics.get(metric, 0.0)
            print(f"  Avg {metric}: {avg_val:.4f} ± {std_val:.4f}")

    print("\n\n" + "=" * 50)
    print("--- All Refinement Ablation CV Results (Averaged ± Std) ---")
    print("=" * 50)
    for mode, results in final_results.items():
        print(f"\nMode: {mode}")
        avg_metrics = results['avg']
        std_metrics = results['std']
        for metric in avg_metrics.keys():
            avg_val = avg_metrics[metric]
            std_val = std_metrics.get(metric, 0.0)
            print(f"  Avg {metric}: {avg_val:.4f} ± {std_val:.4f}")


if __name__ == '__main__':
    main()