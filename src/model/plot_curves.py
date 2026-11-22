"""Loss and validation curves for the Week 3 report (owner: Michaël)

Run: python -m src.model.plot_curves lightgcn nlgcl
Output: results/curves.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS


def main(names=None):
    names = names or sys.argv[1:] or ["lightgcn", "nlgcl"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in names:
        path = RESULTS / f"{name}_history.json"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        hist = json.loads(path.read_text())["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_bpr"] for h in hist], label=f"{name} train")
        axes[0].plot(epochs, [h["val_bpr"] for h in hist], "--", label=f"{name} validation")
        if any(h["train_cl"] for h in hist):
            axes[1].plot(epochs, [h["train_cl"] for h in hist], label=name)
        axes[2].plot(epochs, [h["val_NDCG@10"] for h in hist], label=name)
    axes[0].set_title("BPR loss")
    axes[1].set_title("Contrastive loss (NLGCL)")
    axes[2].set_title("Validation NDCG@10")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "curves.png", dpi=120)
    print("saved", RESULTS / "curves.png")


if __name__ == "__main__":
    main()
