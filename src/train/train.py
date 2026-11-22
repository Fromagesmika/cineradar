"""Train NLGCL or LightGCN on MovieLens (owner: Michaël)

Run: python -m src.train.train --name nlgcl --cl-weight 0.1
     python -m src.train.train --name lightgcn --cl-weight 0
     python -m src.train.train --name toy --toy      (synthetic 1,000 x 500 graph, no data needed)
Saves models/<name>.pt and results/<name>_history.json
"""
import argparse
import json
import time

import numpy as np
import torch

from src.config import MODELS, PROCESSED, RESULTS, SEED
from src.data.split import build_graph, load_split
from src.eval.ranking import ground_truth, mask_matrix, metrics, top_k_lists
from src.model.nlgcl import NLGCL
from src.model.toy import toy_split


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="nlgcl")
    p.add_argument("--cl-weight", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--cl-batch", type=int, default=1024)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--toy", action="store_true", help="train on the synthetic graph from src.model.toy")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train, val, _, meta = toy_split() if args.toy else load_split()
    n_users, n_items = meta["n_users"], meta["n_items"]
    hidden_gems = set(range(n_items)) - set(meta["popular_items"])
    adj = build_graph(train, n_users, n_items)
    model = NLGCL(n_users, n_items, adj, args.dim, args.layers, args.temperature)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    u_all = torch.as_tensor(train["u"].to_numpy(), dtype=torch.long)
    i_all = torch.as_tensor(train["i"].to_numpy(), dtype=torch.long)
    val_u = torch.as_tensor(val["u"].to_numpy(), dtype=torch.long)
    val_i = torch.as_tensor(val["i"].to_numpy(), dtype=torch.long)
    val_neg = torch.as_tensor(rng.integers(0, n_items, len(val)), dtype=torch.long)

    # Validation ranking on a fixed sample of users (speed)
    val_truth = ground_truth(val, n_users)
    val_users = rng.choice(np.unique(val["u"].to_numpy()), size=min(4000, val["u"].nunique()), replace=False)
    train_mask = mask_matrix([train], n_users, n_items)

    history, best, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        perm = torch.randperm(len(u_all))
        sums = {"bpr": 0.0, "cl": 0.0}
        steps = 0
        for b in range(0, len(perm), args.batch):
            idx = perm[b:b + args.batch]
            users, pos = u_all[idx], i_all[idx]
            neg = torch.randint(0, n_items, (len(idx),))
            layers = model.propagate()
            final = model.readout(layers)
            bpr = model.bpr_loss(final, users, pos, neg)
            loss = bpr + args.reg * model.reg_loss(users, pos, neg)
            if args.cl_weight > 0:
                sub = torch.randperm(len(idx))[: args.cl_batch]
                cl = model.contrastive_loss(layers, users[sub], pos[sub])
                loss = loss + args.cl_weight * cl
                sums["cl"] += cl.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums["bpr"] += bpr.item()
            steps += 1

        model.eval()
        with torch.no_grad():
            final = model.readout(model.propagate())
            val_bpr = model.bpr_loss(final, val_u, val_i, val_neg).item()
            user_emb, item_emb = model.split(final)
            topk = top_k_lists(lambda u: user_emb[u] @ item_emb.T, val_users, train_mask, k=50)
            val_metrics = metrics(topk, val_users, val_truth, hidden_gems, n_items)
        record = {
            "epoch": epoch,
            "train_bpr": round(sums["bpr"] / steps, 5),
            "train_cl": round(sums["cl"] / steps, 5),
            "val_bpr": round(val_bpr, 5),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": round(time.time() - start, 1),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (RESULTS / f"{args.name}_history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))

        if val_metrics["NDCG@10"] > best:
            best, bad = val_metrics["NDCG@10"], 0
            torch.save({"state": model.state_dict(), "args": vars(args), "epoch": epoch}, MODELS / f"{args.name}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}, best val NDCG@10 = {best}")
                break
    return history


if __name__ == "__main__":
    main()
