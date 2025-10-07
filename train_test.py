import os
from pathlib import Path

import numpy as np
import torch

from utils import test_eval
from model import *

from dgl.nn import EdgeWeightNorm, GraphConv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import gaussian_kde, entropy, pearsonr, spearmanr, wasserstein_distance
from scipy.spatial.distance import jensenshannon


def to_numpy(x):
    """tensor → numpy"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def average_train_kde(train_kde_list, x_grid):
    densities = []
    for x_vals, kde_vals in train_kde_list:
        d = np.interp(x_grid, x_vals, kde_vals)
        d /= d.sum()  # 归一化
        densities.append(d)
    return np.mean(densities, axis=0)


def compute_js_between(train_kde_list, test_scores, bins=500):
    test_scores = to_numpy(test_scores)

    x_min = min(min(x) for x, _ in train_kde_list)
    x_max = max(max(x) for x, _ in train_kde_list)
    x_min = min(x_min, test_scores.min())
    x_max = max(x_max, test_scores.max())
    x_grid = np.linspace(x_min, x_max, bins)

    train_density = average_train_kde(train_kde_list, x_grid)

    kde = gaussian_kde(test_scores)
    test_density = kde(x_grid)
    test_density /= test_density.sum()

    js_val = jensenshannon(train_density, test_density)
    return float(js_val)


def fuse_scores_with_js(score_features: torch.Tensor,
                        train_query_scores_kde_list,
                        train_mlp_score_kde_list,
                        train_gcn_score_kde_list,
                        bins=500,
                        method="softmax",  # "inverse", "softmax", "diff"
                        alpha=3.0,  # 温度/缩放因子
                        eps=1e-8):
    js_query = compute_js_between(train_query_scores_kde_list, score_features[:, 0], bins=bins)
    js_mlp = compute_js_between(train_mlp_score_kde_list, score_features[:, 1], bins=bins)
    js_gcn = compute_js_between(train_gcn_score_kde_list, score_features[:, 2], bins=bins)

    js_values = {"query": js_query, "mlp": js_mlp, "gcn": js_gcn}
    js_arr = np.array([js_query, js_mlp, js_gcn])

    exp_val = np.exp(-alpha * js_arr)  # alpha
    weights = exp_val / exp_val.sum()

    fused_score = (
            weights[0] * score_features[:, 0] +
            weights[1] * score_features[:, 1] +
            weights[2] * score_features[:, 2]
    )

    return {
        "js_values": {k: round(v, 4) for k, v in js_values.items()},
        "weights": np.round(weights, 4).tolist(),
        "fused_score": fused_score
    }


def select_anomalous_nodes_from_pseudo_labels(pseudo_labels: torch.Tensor,
                                              true_labels,
                                              count=2,
                                              normal_ratio: int = 2,
                                              seed=42):
    N = pseudo_labels.shape[0]
    vote_counts = torch.sum(pseudo_labels, dim=1)  # [N]

    anomaly_indices = torch.where(vote_counts >= count)[0]
    num_anomalies = len(anomaly_indices)

    normal_candidate_indices = torch.where(vote_counts == 0)[0]
    num_normals_to_select = min(len(normal_candidate_indices), num_anomalies * normal_ratio)

    generator = torch.Generator()
    generator.manual_seed(42)
    perm = torch.randperm(len(normal_candidate_indices), generator=generator)
    selected_normal_indices = normal_candidate_indices[perm[:num_normals_to_select]]

    selected_pseudo_indices = torch.cat([anomaly_indices, selected_normal_indices], dim=0)
    final_pseudo_labels = torch.zeros_like(selected_pseudo_indices, dtype=torch.long)
    final_pseudo_labels[:len(anomaly_indices)] = 1

    return final_pseudo_labels, selected_pseudo_indices


def generate_pseudo_labels_for_each_score(score_features, true_labels, sort_order='desc',
                                          num_anomaly=10):
    pseudo_labels = torch.zeros_like(score_features, dtype=torch.long, device=score_features.device)

    if score_features.ndimension() == 1:
        score_features = score_features.unsqueeze(1)

    if sort_order == 'desc':
        sorted_scores, sorted_indices = torch.sort(score_features, dim=0, descending=True)
    else:

        sorted_scores, sorted_indices = torch.sort(score_features, dim=0, descending=False)

    pseudo_labels[sorted_indices[:num_anomaly]] = 1  # 标记得分最大/最小的节点为异常
    return pseudo_labels


def optimize_score_weights(score_features: torch.Tensor,
                           selected_pseudo_indices: torch.Tensor,
                           labels: torch.Tensor,
                           top_k: int = 3,
                           search_grid=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 0.95, 0.99)) -> dict:
    score_norm = score_features
    score_norm_weight = score_norm[selected_pseudo_indices]

    with torch.no_grad():
        if torch.is_tensor(labels):
            score_np = score_norm.cpu().numpy()
            score_norm_weight_np = score_norm_weight.cpu().numpy()
        if torch.is_tensor(labels):
            label_np = labels.cpu().numpy()

    # Step 2: Compute AUROC per dimension
    dim_auc = [roc_auc_score(label_np, score_norm_weight_np[:, i]) for i in range(score_norm_weight_np.shape[1])]
    top_dims = np.argsort(dim_auc)[-top_k:]

    # Step 3: Grid Search over selected dimensions
    grid = search_grid
    best_auc = 0
    best_weights = None

    for weights in itertools.product(grid, repeat=top_k):
        w = np.array(weights)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        fused = (score_norm_weight_np[:, top_dims] * w).sum(axis=1)
        auc = roc_auc_score(label_np, fused)
        if auc > best_auc:
            best_auc = auc
            best_weights = w

    fused_score_tensor = torch.tensor(
        (score_np[:, top_dims] * best_weights).sum(axis=1),
        device=score_features.device,
        dtype=score_features.dtype
    )

    return {
        'best_auc': round(best_auc, 4),
        'best_weights': np.round(best_weights, 4).tolist(),
        'best_dims': top_dims.tolist(),
        'fused_score': fused_score_tensor
    }


def load_out_t(out_t_dir, name):
    return torch.from_numpy(np.load(out_t_dir.joinpath(name))["arr_0"])


def tam_score_euclidean(adj, node_feat):
    adj = adj.coalesce()
    row, col = adj.indices()
    N = node_feat.size(0)
    dist = torch.norm(node_feat[row] - node_feat[col], p=2, dim=1)
    score_sum = torch.zeros(N, device=node_feat.device)
    degree = torch.zeros(N, device=node_feat.device)
    score_sum.index_add_(0, row, dist)
    degree.index_add_(0, row, torch.ones_like(dist))
    anomaly_score = score_sum / (degree + 1e-8)
    anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)
    return anomaly_score


def tam_score_abs(adj, node_feat):
    adj = adj.coalesce()
    row, col = adj.indices()
    N = node_feat.size(0)

    dist = torch.norm(node_feat[row] - node_feat[col], p=1, dim=1)

    score_sum = torch.zeros(N, device=node_feat.device)
    degree = torch.zeros(N, device=node_feat.device)

    score_sum.index_add_(0, row, dist)
    degree.index_add_(0, row, torch.ones_like(dist))

    anomaly_score = score_sum / (degree + 1e-8)
    anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-8)

    return anomaly_score


def tam_cos_score(adj_matrix, feature):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    message = torch.sum(sim_matrix, 1)

    message = message * r_inv

    return - torch.sum(message), message


def tam_combined_scores(adj, node_feat):
    if not adj.is_sparse:
        adj = adj.to_sparse()

    adj = adj.coalesce()
    row, col = adj.indices()
    N = node_feat.size(0)
    device = node_feat.device

    # === L2 Score ===
    dist_l2 = torch.norm(node_feat[row] - node_feat[col], p=2, dim=1)
    score_sum_l2 = torch.zeros(N, device=device)
    degree_l2 = torch.zeros(N, device=device)
    score_sum_l2.index_add_(0, row, dist_l2)
    degree_l2.index_add_(0, row, torch.ones_like(dist_l2))
    score_l2 = score_sum_l2 / (degree_l2 + 1e-8)
    score_l2 = (score_l2 - score_l2.min()) / (score_l2.max() - score_l2.min() + 1e-8)

    # === L1 Score ===
    dist_l1 = torch.norm(node_feat[row] - node_feat[col], p=1, dim=1)
    score_sum_l1 = torch.zeros(N, device=device)
    degree_l1 = torch.zeros(N, device=device)
    score_sum_l1.index_add_(0, row, dist_l1)
    degree_l1.index_add_(0, row, torch.ones_like(dist_l1))
    score_l1 = score_sum_l1 / (degree_l1 + 1e-8)
    score_l1 = (score_l1 - score_l1.min()) / (score_l1.max() - score_l1.min() + 1e-8)

    # === Cosine Score ===
    feat_norm = node_feat / (torch.norm(node_feat, dim=-1, keepdim=True) + 1e-8)
    sim = (feat_norm[row] * feat_norm[col]).sum(dim=1)  # cosine similarity on edges

    score_sum_cos = torch.zeros(N, device=device)
    degree_cos = torch.zeros(N, device=device)
    score_sum_cos.index_add_(0, row, sim)
    degree_cos.index_add_(0, row, torch.ones_like(sim))
    score_cos = score_sum_cos / (degree_cos + 1e-8)
    score_cos = (score_cos - score_cos.min()) / (score_cos.max() - score_cos.min() + 1e-8)

    # score_cos = 1 - score_cos
    loss_cos = -torch.sum(score_cos)

    return score_l2, score_l1, score_cos, loss_cos


import numpy as np
import torch
from numba import njit
import math


@njit(cache=True, fastmath=True)
def _accum_l2(row, col, feat):
    N, D = feat.shape
    score_sum = np.zeros(N, dtype=np.float32)
    degree = np.zeros(N, dtype=np.float32)
    for i in range(row.shape[0]):
        r = row[i]
        c = col[i]
        s = 0.0
        for d in range(D):
            diff = feat[r, d] - feat[c, d]
            s += diff * diff
        score_sum[r] += math.sqrt(s)
        degree[r] += 1.0
    return score_sum, degree


@njit(cache=True, fastmath=True)
def _accum_l1(row, col, feat):
    N, D = feat.shape
    score_sum = np.zeros(N, dtype=np.float32)
    degree = np.zeros(N, dtype=np.float32)
    for i in range(row.shape[0]):
        r = row[i]
        c = col[i]
        s = 0.0
        for d in range(D):
            s += abs(feat[r, d] - feat[c, d])
        score_sum[r] += s
        degree[r] += 1.0
    return score_sum, degree


@njit(cache=True, fastmath=True)
def _accum_cos(row, col, feat_norm):
    N, D = feat_norm.shape
    score_sum = np.zeros(N, dtype=np.float32)
    degree = np.zeros(N, dtype=np.float32)
    for i in range(row.shape[0]):
        r = row[i]
        c = col[i]
        s = 0.0
        for d in range(D):
            s += feat_norm[r, d] * feat_norm[c, d]
        score_sum[r] += s
        degree[r] += 1.0
    return score_sum, degree


def tam_combined_scores_cpu(adj: torch.Tensor, node_feat: torch.Tensor):
    eps = 1e-8
    orig_device = node_feat.device

    if not adj.is_sparse:
        adj = adj.to_sparse()
    adj = adj.coalesce()
    row, col = adj.indices()  # [E], [E]

    feat_np = node_feat.detach().cpu().to(torch.float32).numpy()
    row_np = row.cpu().to(torch.int64).numpy()
    col_np = col.cpu().to(torch.int64).numpy()

    # ---- L2 / L1 ----
    l2_sum, l2_deg = _accum_l2(row_np, col_np, feat_np)
    l1_sum, l1_deg = _accum_l1(row_np, col_np, feat_np)

    score_l2 = l2_sum / (l2_deg + eps)
    score_l1 = l1_sum / (l1_deg + eps)

    def _minmax(x):
        xmin = x.min()
        xmax = x.max()
        return (x - xmin) / (xmax - xmin + eps) if xmax > xmin else np.zeros_like(x, dtype=np.float32)

    score_l2 = _minmax(score_l2.astype(np.float32))
    score_l1 = _minmax(score_l1.astype(np.float32))

    norm = np.linalg.norm(feat_np, axis=1, keepdims=True) + eps
    feat_norm_np = feat_np / norm

    cos_sum, cos_deg = _accum_cos(row_np, col_np, feat_norm_np)
    score_cos = cos_sum / (cos_deg + eps)
    score_cos = _minmax(score_cos.astype(np.float32))

    loss_cos = -score_cos.sum(dtype=np.float32)

    score_l2 = torch.from_numpy(score_l2).to(orig_device)
    score_l1 = torch.from_numpy(score_l1).to(orig_device)
    score_cos = torch.from_numpy(score_cos).to(orig_device)
    loss_cos = torch.tensor(loss_cos, device=orig_device)

    return score_l2, score_l1, score_cos, loss_cos


def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.

    message = torch.sum(sim_matrix, 1)

    message = message * r_inv

    return - torch.sum(message), message


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


class my_GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(my_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 2 * h_feats)
        self.conv2 = GraphConv(2 * h_feats, h_feats)

        self.fc1 = nn.Linear(h_feats, h_feats, bias=False)
        self.fc2 = nn.Linear(h_feats, h_feats, bias=False)

        # self.param_init()
        # self.fc1 = nn.Linear(h_feats, 1, bias=False)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)

        return h

    def get_final_predict(self, h):
        return torch.sigmoid(self.fc1(h))

    def param_init(self):
        nn.init.xavier_normal_(self.conv1.weight, gain=1.414)
        nn.init.xavier_normal_(self.conv2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)


def save_result_codebook(codebook, quantized, train_data):
    output_dir = Path.cwd().joinpath(
        "output",
        f"{train_data.name}"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    codebook = codebook.detach().cpu().numpy()
    quantized = quantized.detach().cpu().numpy()
    np.savez(output_dir.joinpath("codebook"), codebook)
    np.savez(output_dir.joinpath("node_emb_with_I_emebding"), quantized)


def load_result_codebook(train_data):
    output_dir = Path.cwd().joinpath(
        "output_score",
        f"{train_data.name}"
    )

    query_scores = load_out_t(output_dir, 'query_scores.npz')
    mlp_score_all = load_out_t(output_dir, 'mlp_score_all.npz')
    gcn_score_all = load_out_t(output_dir, 'gcn_score_all.npz')

    return query_scores, mlp_score_all, gcn_score_all


def save_result_anomaly_score(query_scores, mlp_score_all, gcn_score_all, train_data):
    output_dir = Path.cwd().joinpath(
        "output_score",
        f"{train_data.name}"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    query_scores = query_scores.detach().cpu().numpy()
    mlp_score_all = mlp_score_all.detach().cpu().numpy()
    gcn_score_all = gcn_score_all.detach().cpu().numpy()

    np.savez(output_dir.joinpath("query_scores"), query_scores)
    np.savez(output_dir.joinpath("mlp_score_all"), mlp_score_all)
    np.savez(output_dir.joinpath("gcn_score_all"), gcn_score_all)


def compute_kde_distribution(scores, bins=500):
    if not isinstance(scores, np.ndarray):
        scores = scores.detach().cpu().numpy()

    kde = gaussian_kde(scores)
    x_vals = np.linspace(scores.min(), scores.max(), bins)
    density = kde(x_vals)
    density /= density.sum()

    return x_vals, density


class TA_GGAD_Detector:
    def __init__(self, train_config, model_config, data, args):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.args = args

        self.device = train_config['device']
        self.model = ARC(args, **model_config).to(train_config['device'])

        self.model_dir = "./saved_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.train_dataset_names = [d.name for d in data['train']]

    def get_model_path(self, seed):
        return os.path.join(self.model_dir,
                            f"{self.model_config['model']}_seed{seed}_h_feats{self.model_config['h_feats']}.pth")

    def model_exists(self, seed):
        return os.path.exists(self.get_model_path(seed))

    def save_model(self, seed):

        save_data = {'state_dict': self.model.state_dict(), 'model_config': self.model_config,
                     'train_dataset_names': self.train_dataset_names}
        torch.save(save_data, self.get_model_path(seed))
        print(f"Model saved for seed {seed}")

    def load_model(self, seed):
        if not self.model_exists(seed):
            raise FileNotFoundError(f"No model found for seed {seed}")

        save_data = torch.load(self.get_model_path(seed))

        if set(save_data['train_dataset_names']) != set(self.train_dataset_names):
            raise ValueError(
                f"Training datasets mismatch. Saved: {save_data['train_dataset_names']}, Current: {self.train_dataset_names}")

        self.model.load_state_dict(save_data['state_dict'])
        print(f"Model loaded for seed {seed}")

    def train(self, args):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_config['lr'],
                                      weight_decay=self.model_config['weight_decay'])

        for e in range(self.train_config['epochs']):
            for didx, train_data in enumerate(self.data['train']):

                self.model.train()
                train_graph = self.data['train'][didx].graph.to(self.device)
                residual_embed, loss_code, quantized, node_mlp_embd, codebook = self.model(train_graph, train_data)

                loss = self.model.get_train_loss(residual_embed, quantized, codebook,
                                                 train_graph.ano_labels,
                                                 self.model_config['num_prompt'])

                cut_adj = train_graph.dgl_cut_graph.adj().to(self.device)
                score_l2_mlp, score_l1_mlp, score_cos_mlp, loss_cos_mlp = tam_combined_scores(cut_adj, node_mlp_embd)
                score_l2_gcn, score_l1_gcn, score_cos_gcn, loss_cos_gcn = tam_combined_scores(cut_adj, quantized)

                query_scores = self.model.get_test_score(residual_embed, codebook,
                                                         train_graph.shot_mask,
                                                         train_graph.ano_labels)

                mlp_score_all = score_l2_mlp + score_l1_mlp + score_cos_mlp
                gcn_score_all = score_l2_gcn + score_l1_gcn + score_cos_gcn

                loss = loss + loss_code.squeeze() + loss_cos_mlp + loss_cos_gcn

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if e % 20 == 0:
                    print(f"current epoch {e}")

                if e == self.train_config['epochs'] - 1:
                    save_result_codebook(codebook, quantized, train_data)
                    save_result_anomaly_score(query_scores, mlp_score_all, gcn_score_all, train_data)

        print('Finish Training for {} epochs!'.format(self.train_config['epochs']))
        max_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"max_gpu_usage{max_memory_MB:.2f} GB")

    def test(self, args):
        # Evaluation
        test_score_list = {}
        test_score_list_MOE = {}
        self.model.eval()
        # self.GCN_model.eval()

        # codebook_sum = None
        codebook_list = []
        train_query_scores_list = []
        train_mlp_score_list = []
        train_gcn_score_list = []
        train_query_scores_kde_list = []
        train_mlp_score_kde_list = []
        train_gcn_score_kde_list = []

        for didx, train_data in enumerate(self.data['train']):
            output_dir = Path.cwd().joinpath(
                "output",
                f"{train_data.name}"
            )

            codebook_1 = load_out_t(output_dir, 'codebook.npz')
            codebook_list.append(codebook_1)
            query_scores_train, mlp_score_all, gcn_score_all = load_result_codebook(train_data)
            train_query_scores_list.append(query_scores_train)
            train_mlp_score_list.append(mlp_score_all)
            train_gcn_score_list.append(gcn_score_all)
            x_q, kde_q = compute_kde_distribution(query_scores_train)
            x_m, kde_m = compute_kde_distribution(mlp_score_all)
            x_g, kde_g = compute_kde_distribution(gcn_score_all)

            train_query_scores_kde_list.append((x_q, kde_q))
            train_mlp_score_kde_list.append((x_m, kde_m))
            train_gcn_score_kde_list.append((x_g, kde_g))

        final_codebook = torch.cat(codebook_list, dim=0).to(self.train_config['device'])

        for didx, test_data in enumerate(self.data['test']):
            test_graph = test_data.graph.to(self.train_config['device'])
            labels = test_graph.ano_labels
            shot_mask = test_graph.shot_mask.bool()
            query_labels = labels[~shot_mask].to(self.train_config['device'])

            num_anomaly = int(torch.sum(query_labels == 1).data)

            residual_embed, _, quantized, node_mlp_embd, codebook = self.model(test_graph, test_data)

            output_dir = Path.cwd().joinpath(
                "output",
                f"{test_data.name}"
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            query_scores = self.model.get_test_score(residual_embed, final_codebook,
                                                     test_graph.shot_mask,
                                                     test_graph.ano_labels)

            cut_adj = test_graph.dgl_cut_graph.adj().to(self.device)
            if test_data.name in ['DGgraph-fin', 'tfinance']:
                score_l2_mlp, score_l1_mlp, score_cos_mlp, _ = tam_combined_scores_cpu(cut_adj, node_mlp_embd)
                score_l2_gcn, score_l1_gcn, score_cos_gcn, _ = tam_combined_scores_cpu(cut_adj, quantized)
            else:
                score_l2_mlp, score_l1_mlp, score_cos_mlp, _ = tam_combined_scores(cut_adj, node_mlp_embd)
                score_l2_gcn, score_l1_gcn, score_cos_gcn, _ = tam_combined_scores(cut_adj, quantized)

            mlp_score_all = score_l2_mlp + score_l1_mlp + score_cos_mlp
            gcn_score_all = score_l2_gcn + score_l1_gcn + score_cos_gcn

            affinity_mlp = np.array(mlp_score_all.cpu().detach())
            affinity_gcn = np.array(gcn_score_all.cpu().detach())

            final_affinity_mlp = 1 - normalize_score(affinity_mlp)
            final_affinity_gcn = 1 - normalize_score(affinity_gcn)

            final_affinity_mlp = torch.FloatTensor(final_affinity_mlp).to(self.device)
            final_affinity_gcn = torch.FloatTensor(final_affinity_gcn).to(self.device)
            lamda = test_graph.l

            query_scores_mlp = (1 - lamda) * query_scores + lamda * final_affinity_mlp[~shot_mask]
            query_scores_gcn = (1 - lamda) * query_scores + lamda * final_affinity_gcn[~shot_mask]

            score_features = torch.stack([
                query_scores,
                query_scores_mlp,
                query_scores_gcn,
            ], dim=1)

            js_fused_res = fuse_scores_with_js(
                score_features,
                train_query_scores_kde_list,
                train_mlp_score_kde_list,
                train_gcn_score_kde_list
            )

            js_fused_score = js_fused_res["fused_score"]

            score_features = torch.cat([score_features, js_fused_score.unsqueeze(1)], dim=1)

            pseudo_labels_query = generate_pseudo_labels_for_each_score(query_scores, query_labels,
                                                                        sort_order='desc',
                                                                        num_anomaly=num_anomaly)

            pseudo_labels_query_mlp = generate_pseudo_labels_for_each_score(query_scores_mlp, query_labels,
                                                                            sort_order='desc',
                                                                            num_anomaly=num_anomaly)

            pseudo_labels_query_gcn = generate_pseudo_labels_for_each_score(query_scores_gcn, query_labels,
                                                                            sort_order='desc',
                                                                            num_anomaly=num_anomaly)

            pseudo_labels_all = torch.stack([
                pseudo_labels_query,
                pseudo_labels_query_mlp,
                pseudo_labels_query_gcn
            ], dim=1)

            final_pseudo_labels, selected_pseudo_indices = select_anomalous_nodes_from_pseudo_labels(pseudo_labels_all,
                                                                                                     query_labels,
                                                                                                     args.count_node,
                                                                                                     args.normal_ratio,
                                                                                                     )
            result = optimize_score_weights(score_features, selected_pseudo_indices, final_pseudo_labels)
            print("✅ Best AUROC:", result['best_auc'])
            print("🎯 Optimal Weights:", result['best_weights'])
            print("📊 Selected Dims:", result['best_dims'])
            fused_score = result['fused_score']
            query_scores = torch.mean(score_features, dim=1)
            test_score_MOE = test_eval(query_labels, fused_score)
            test_data_name = self.train_config['testdsets'][didx]
            test_score_list_MOE[test_data_name] = {
                'AUROC': test_score_MOE['AUROC'],
                'AUPRC': test_score_MOE['AUPRC'],
            }
            print("\n\n\n")
        return test_score_list_MOE
