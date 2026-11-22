"""NLGCL model (owner: Michaël)

Paper: Xu et al., "NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning
for Recommendation", RecSys 2025

LightGCN propagation e^(l+1) = A_hat e^(l). A node at layer l and its neighbors at layer l+1
are positive pairs in an InfoNCE loss, so no graph augmentation is needed
Total loss = BPR + cl_weight * contrastive + reg. With cl_weight = 0 the model is LightGCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricSparseMM(torch.autograd.Function):
    """A @ X for a symmetric sparse A. The gradient is A^T @ g = A @ g, so the backward
    reuses the CSR matrix. About 10x faster on CPU than torch.sparse.mm's own backward"""

    @staticmethod
    def forward(ctx, adj, x):
        ctx.adj = adj
        return adj @ x

    @staticmethod
    def backward(ctx, grad):
        return None, ctx.adj @ grad


class NLGCL(nn.Module):
    def __init__(self, n_users: int, n_items: int, adj: torch.Tensor, dim: int = 64,
                 n_layers: int = 3, temperature: float = 0.2):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        # The normalized adjacency D^-1/2 A D^-1/2 of an undirected graph is symmetric
        self.adj = adj.to_sparse_csr() if adj.layout != torch.sparse_csr else adj
        self.n_layers = n_layers
        self.temperature = temperature
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self):
        layers = [self.embedding.weight]
        for _ in range(self.n_layers):
            layers.append(SymmetricSparseMM.apply(self.adj, layers[-1]))
        return layers

    @staticmethod
    def readout(layers):
        return torch.stack(layers).mean(dim=0)

    def split(self, emb):
        return emb[: self.n_users], emb[self.n_users:]

    def bpr_loss(self, final, users, pos, neg):
        user_emb, item_emb = self.split(final)
        u, p, n = user_emb[users], item_emb[pos], item_emb[neg]
        return -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1)).mean()

    def reg_loss(self, users, pos, neg):
        ego = self.embedding.weight
        u, p, n = ego[users], ego[self.n_users + pos], ego[self.n_users + neg]
        return (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / (2 * len(users))

    def _info_nce(self, anchor, positive):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.T / self.temperature
        labels = torch.arange(len(anchor), device=anchor.device)
        return F.cross_entropy(logits, labels)

    def contrastive_loss(self, layers, users, pos):
        """Neighbor-layer contrast: user u at layer l vs liked film i at layer l+1, and the reverse"""
        loss = 0.0
        for layer in range(self.n_layers):
            cur, nxt = layers[layer], layers[layer + 1]
            loss = loss + self._info_nce(cur[users], nxt[self.n_users + pos])
            loss = loss + self._info_nce(cur[self.n_users + pos], nxt[users])
        return loss / (2 * self.n_layers)

    @torch.no_grad()
    def final_embeddings(self):
        return self.split(self.readout(self.propagate()))
