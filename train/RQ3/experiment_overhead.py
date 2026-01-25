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
from collections import Counter
import os
import time
import psutil

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
    OPCODE_MIN_FREQ = 1
    DFG_MIN_FREQ = 4
    N_SPLITS = 10


def flatten_corpus_iterative(nested_list):
    if not isinstance(nested_list, list): return [nested_list]
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
            if count >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_built = True
        print(f"Vocab Size: {len(self.token_to_id)}")

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
            op_seq = flatten_corpus_iterative(ast.literal_eval(row['oplist']))
        except:
            op_seq = []
        op_tensor = torch.tensor(self.op_tokenizer.tokens_to_ids(op_seq), dtype=torch.long)
        try:
            cfg_graph = self._create_cfg_graph(json.loads(row['basicblocks']))
        except:
            cfg_graph = self._create_empty_graph(True)
        try:
            dfg_nodes = flatten_corpus_iterative(json.loads(row['node_feature']))
            dfg_edges = json.loads(row['edge_index'])
            dfg_graph = self._create_dfg_graph(dfg_nodes, dfg_edges)
        except:
            dfg_graph = self._create_empty_graph(False)
        return op_tensor, cfg_graph, dfg_graph, torch.tensor(row['label'], dtype=torch.float)

    def _create_cfg_graph(self, full_cfg_json):
        if 'basicblocks' not in full_cfg_json or not full_cfg_json['basicblocks']: return self._create_empty_graph(True)
        blocks = full_cfg_json['basicblocks']
        if not isinstance(blocks, dict): return self._create_empty_graph(True)
        b_ids = sorted(blocks.keys(), key=int)
        id_map = {bid: i for i, bid in enumerate(b_ids)}
        feats, edges = [], []
        for bid in b_ids:
            ops = [i['opname'] for i in blocks[bid].get('instructions', [])]
            feats.append(torch.tensor(self.op_tokenizer.tokens_to_ids(ops), dtype=torch.long))
            curr = id_map[bid]
            for succ in blocks[bid].get('successors', []):
                if str(succ) in id_map: edges.append([curr, id_map[str(succ)]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0,
                                                                                                      dtype=torch.long)
        return Data(x=feats, edge_index=edge_index)

    def _create_dfg_graph(self, nodes, edges):
        if not nodes: return self._create_empty_graph(False)
        x = torch.tensor(self.dfg_tokenizer.tokens_to_ids(nodes), dtype=torch.long).unsqueeze(1)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0,
                                                                                                      dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def _create_empty_graph(self, is_cfg):
        x = [] if is_cfg else torch.empty(0, 1, dtype=torch.long)
        return Data(x=x, edge_index=torch.empty(2, 0, dtype=torch.long))


def collate_fn(batch):
    ops, cfgs, dfgs, lbls = zip(*batch)
    return pad_sequence(ops, batch_first=True), Batch.from_data_list(cfgs), Batch.from_data_list(dfgs), torch.stack(
        lbls)


class OpcodeLSTM(nn.Module):
    def __init__(self, emb, h_dim):
        super().__init__()
        self.embedding = emb
        self.lstm = nn.LSTM(emb.embedding_dim, h_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(h_dim * 2, h_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embedding(x))
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))


class BlockEncoder(nn.Module):
    def __init__(self, emb, h_dim):
        super().__init__()
        self.embedding = emb
        self.gru = nn.GRU(emb.embedding_dim, h_dim, batch_first=True)

    def forward(self, x):
        _, h = self.gru(self.embedding(x.unsqueeze(0)))
        return h.squeeze(0).squeeze(0)


class GraphGNN(nn.Module):
    def __init__(self, din, dh, dout):
        super().__init__()
        self.conv1 = GCNConv(din, dh)
        self.conv2 = GCNConv(dh, dout)
        self.relu = nn.ReLU()
        self.out_channels = dout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.size(0) == 0: return torch.zeros(data.num_graphs, self.out_channels, device=edge_index.device)
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)


class GatedFusion(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.gate = nn.Linear(din * 3, 3)
        self.out = nn.Linear(din, dout)
        self.tanh = nn.Tanh()

    def forward(self, op, cfg, dfg):
        g = torch.softmax(self.gate(torch.cat([op, cfg, dfg], dim=1)), dim=1)
        return self.tanh(self.out(g[:, 0:1] * op + g[:, 1:2] * cfg + g[:, 2:3] * dfg))


class PonziDetector(nn.Module):
    def __init__(self, op_sz, dfg_sz, cfg):
        super().__init__()
        self.cfg = cfg
        self.op_emb = nn.Embedding(op_sz, cfg.EMBEDDING_DIM, padding_idx=0)
        self.var_emb = nn.Embedding(dfg_sz, cfg.EMBEDDING_DIM, padding_idx=0)
        self.blk_enc = BlockEncoder(self.op_emb, cfg.GNN_HIDDEN_DIM)
        self.op_enc = OpcodeLSTM(self.op_emb, cfg.LSTM_HIDDEN_DIM)
        self.cfg_gnn = GraphGNN(cfg.GNN_HIDDEN_DIM, cfg.GNN_HIDDEN_DIM, cfg.GNN_HIDDEN_DIM)
        self.dfg_gnn = GraphGNN(cfg.EMBEDDING_DIM, cfg.GNN_HIDDEN_DIM, cfg.GNN_HIDDEN_DIM)
        self.drop = nn.Dropout(0.5)
        self.fusion = GatedFusion(cfg.LSTM_HIDDEN_DIM, cfg.FUSION_DIM)
        self.cls = nn.Sequential(nn.Linear(cfg.FUSION_DIM, cfg.FUSION_DIM // 2), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(cfg.FUSION_DIM // 2, 1))

    def forward(self, op, cfg, dfg):
        op_v = self.op_enc(op)

        cfg_nodes = []
        if hasattr(cfg, 'x') and isinstance(cfg.x, list):
            for n in cfg.x:
                if isinstance(n, torch.Tensor) and n.numel() > 0:
                    cfg_nodes.append(self.blk_enc(n.to(op.device)))
        if cfg_nodes:
            cfg.x = torch.stack(cfg_nodes)
        else:
            cfg.x = torch.empty(0, self.cfg.GNN_HIDDEN_DIM, device=op.device)
        cfg_v = self.cfg_gnn(cfg)

        if dfg.x.numel() > 0:
            dfg.x = self.var_emb(dfg.x.squeeze(-1))
        else:
            dfg.x = torch.empty(0, self.cfg.EMBEDDING_DIM, device=op.device)
        dfg_v = self.dfg_gnn(dfg)

        return torch.sigmoid(self.cls(self.fusion(self.drop(op_v), self.drop(cfg_v), self.drop(dfg_v))).squeeze(1))


def get_sys_mem():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def train_epoch(model, loader, opt, crit, dev):
    model.train()
    loss_sum = 0
    sys_mems = []
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device=dev)
        torch.cuda.empty_cache()

    for ops, cfgs, dfgs, lbls in loader:
        ops, lbls = ops.to(dev), lbls.to(dev)
        cfgs, dfgs = cfgs.to(dev), dfgs.to(dev)
        opt.zero_grad()
        loss = crit(model(ops, cfgs, dfgs), lbls)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        sys_mems.append(get_sys_mem())

    gpu_peak = torch.cuda.max_memory_allocated(device=dev) / (1024 ** 2) if dev.type == 'cuda' else 0
    sys_peak = np.max(sys_mems) if sys_mems else 0
    return loss_sum / len(loader), gpu_peak, sys_peak


def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum = 0
    preds, lbls = [], []
    with torch.no_grad():
        for ops, cfgs, dfgs, y in loader:
            ops, y = ops.to(dev), y.to(dev)
            cfgs, dfgs = cfgs.to(dev), dfgs.to(dev)
            out = model(ops, cfgs, dfgs)
            loss_sum += crit(out, y).item()
            preds.extend(out.cpu().numpy())
            lbls.extend(y.cpu().numpy())

    preds, lbls = np.array(preds), np.array(lbls)
    if len(np.unique(lbls)) < 2: return 0, 0, 0, 0, 0, 0, 0, 0

    best_f1, best_th = 0, 0.5
    for th in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(lbls, (preds > th).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, th

    p_bin = (preds > best_th).astype(int)
    return loss_sum / len(loader), roc_auc_score(lbls, preds), average_precision_score(lbls,
                                                                                       preds), best_f1, precision_score(
        lbls, p_bin), recall_score(lbls, p_bin), accuracy_score(lbls, p_bin), best_th


def main():
    cfg = Config()
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    gpu_total = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 2) if dev.type == 'cuda' else 0
    sys_total = psutil.virtual_memory().total / (1024 ** 2)

    try:
        df = pd.merge(pd.read_csv(cfg.OPCODE_FILE).rename(columns={'opcodes': 'oplist'}),
                      pd.read_csv(cfg.CFG_FILE).rename(columns={'nodes': 'basicblocks'}), on='address', how='outer')
        df = pd.merge(df, pd.read_csv(cfg.DFG_FILE), on='address', how='outer')
        df = pd.merge(df, pd.read_csv(cfg.LABEL_FILE), on='address', how='outer')
        df.dropna(subset=['oplist', 'basicblocks', 'node_feature', 'edge_index', 'label'], inplace=True)
        df.reset_index(drop=True, inplace=True)
    except Exception as e:
        return print(e)
    print(f"Data Loaded: {len(df)}")

    kfold = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED)
    res, overhead = [], []

    for fold, (tr_idx, te_idx) in enumerate(kfold.split(df, df['label'])):
        print(f"\n=== Fold {fold + 1} ===")
        train_val, test = df.iloc[tr_idx], df.iloc[te_idx]
        train, val = train_test_split(train_val, test_size=cfg.VAL_SIZE / (1 - cfg.TEST_SIZE),
                                      stratify=train_val['label'], random_state=cfg.RANDOM_SEED)

        op_tok = Tokenizer(min_freq=cfg.OPCODE_MIN_FREQ)
        op_tok.build_vocab(flatten_corpus_iterative([ast.literal_eval(s) for s in train['oplist']]))
        dfg_tok = Tokenizer(min_freq=cfg.DFG_MIN_FREQ)
        dfg_tok.build_vocab(flatten_corpus_iterative([json.loads(s) for s in train['node_feature']]))

        dl_args = {'batch_size': cfg.BATCH_SIZE, 'collate_fn': collate_fn}
        tr_dl = DataLoader(PonziDataset(train, op_tok, dfg_tok), shuffle=True, **dl_args)
        va_dl = DataLoader(PonziDataset(val, op_tok, dfg_tok), shuffle=False, **dl_args)
        te_dl = DataLoader(PonziDataset(test, op_tok, dfg_tok), shuffle=False, **dl_args)

        model = PonziDetector(op_tok.get_vocab_size(), dfg_tok.get_vocab_size(), cfg).to(dev)
        opt = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=3)
        crit = nn.BCELoss()

        best_f1, patience, best_path = 0, 0, f'best_{fold}.pth'
        tr_pk_gpu, tr_pk_sys, t_start = 0, 0, time.time()

        for ep in range(cfg.EPOCHS):
            loss, g_m, s_m = train_epoch(model, tr_dl, opt, crit, dev)
            tr_pk_gpu = max(tr_pk_gpu, g_m)
            tr_pk_sys = max(tr_pk_sys, s_m)
            _, _, _, f1, _, _, _, _ = evaluate(model, va_dl, crit, dev)
            sched.step(f1)
            if f1 > best_f1:
                best_f1, patience = f1, 0; torch.save(model.state_dict(), best_path)
            else:
                patience += 1;
            if patience >= cfg.EARLY_STOPPING_PATIENCE: break

        mtime = time.time() - t_start
        model_sz = os.path.getsize(best_path) / (1024 ** 2)
        model.load_state_dict(torch.load(best_path))

        _, auc, _, f1, _, _, acc, _ = evaluate(model, te_dl, crit, dev)
        res.append({'AUC': auc, 'F1': f1, 'Acc': acc})
        print(f" Test: AUC {auc:.4f} F1 {f1:.4f}")

        model.eval()
        t_det = time.time()
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device=dev)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        sys_mems_det = []
        with torch.no_grad():
            for ops, cfgs, dfgs, _ in tqdm(te_dl, desc="Detect"):
                ops, cfgs, dfgs = ops.to(dev), cfgs.to(dev), dfgs.to(dev)
                _ = model(ops, cfgs, dfgs)
                sys_mems_det.append(get_sys_mem())
        if dev.type == 'cuda': torch.cuda.synchronize()
        dtime = (time.time() - t_det) / len(test) * 1000
        dt_pk_gpu = torch.cuda.max_memory_allocated(device=dev) / (1024 ** 2) if dev.type == 'cuda' else 0
        dt_pk_sys = np.max(sys_mems_det) if sys_mems_det else 0

        overhead.append({
            'MTime_s': mtime, 'DTime_ms': dtime,
            'Tr_GPU_MB': tr_pk_gpu, 'Tr_GPU_%': tr_pk_gpu / gpu_total * 100,
            'Dt_GPU_MB': dt_pk_gpu, 'Dt_GPU_%': dt_pk_gpu / gpu_total * 100,
            'Tr_RAM_MB': tr_pk_sys, 'Tr_RAM_%': tr_pk_sys / sys_total * 100,
            'Dt_RAM_MB': dt_pk_sys, 'Dt_RAM_%': dt_pk_sys / sys_total * 100,
            'Model_MB': model_sz
        })
        os.remove(best_path)

    print("\n=== Performance ===")
    print(pd.DataFrame(res).describe().loc[['mean', 'std']])
    print("\n=== Overhead ===")
    print(pd.DataFrame(overhead).describe().loc[['mean', 'std']])


if __name__ == '__main__':
    main()