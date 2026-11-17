"""NLGCL model (owner: Michaël).

Paper: Xu et al., "NLGCL: Naturally Existing Neighbor Layers Graph Contrastive
Learning for Recommendation", RecSys 2025.

Idea: LightGCN propagation  e^(l+1) = A_hat @ e^(l).
A node at layer l and its neighbors at layer l+1 are positive pairs
for a contrastive (InfoNCE) loss. Total loss = BPR + lambda * CL + reg.
"""
import torch
import torch.nn as nn


class NLGCL(nn.Module):
    def __init__(self, n_users: int, n_items: int, adj: torch.Tensor,
                 dim: int = 64, n_layers: int = 3, temperature: float = 0.2):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.adj = adj                      # sparse (n_users + n_items) x (n_users + n_items)
        self.n_layers = n_layers
        self.temperature = temperature
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self):
        """Return the list of embeddings at each layer [e0, e1, ..., eL]."""
        layers = [self.embedding.weight]
        for _ in range(self.n_layers):
            layers.append(torch.sparse.mm(self.adj, layers[-1]))
        return layers

    def forward(self):
        """Final user and item embeddings = mean of all layers."""
        final = torch.stack(self.propagate()).mean(dim=0)
        return final[: self.n_users], final[self.n_users:]

    def bpr_loss(self, users, pos_items, neg_items):
        # TODO (Michaël)
        raise NotImplementedError

    def contrastive_loss(self, users, pos_items):
        """TODO (Michaël): InfoNCE between layer l of each node and layer l+1
        of its neighbors (see section 3 of the paper)."""
        raise NotImplementedError
