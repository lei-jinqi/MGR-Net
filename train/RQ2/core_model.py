# core_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import ast
import json
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    OPCODE_FILE = '../dataset/lists/filteredcfglists.csv'
    CFG_FILE = '../dataset/blocks/filteredcfgs.csv'
    DFG_FILE = '../dataset/dfg.csv'
    LABEL_FILE = '../dataset/label.csv'

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
    DFG_MIN_FREQ = 5


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
        print(f"词汇表构建完成 (>= {self.min_freq} 词频): {self.get_vocab_size()} tokens")

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


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for op_data, cfg_data, dfg_data, labels in tqdm(dataloader, desc="训练中"):
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
        for op_data, cfg_data, dfg_data, labels in tqdm(dataloader, desc="评估中"):
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
    return total_loss / len(dataloader), auc, best_f1, precision, recall, best_threshold