"""Training loop (owner: Rayan).

TODO (Rayan):
1. Load data/processed/train.parquet and val.parquet, build the graph (src.data.split.build_graph)
2. Sample (user, positive film, negative film) triplets for BPR
3. Train NLGCL with Adam, evaluate NDCG@10 on validation after each epoch
4. Stop when validation NDCG stops improving, save the best model to models/nlgcl.pt
5. Save the loss and NDCG curves (needed for the Week 3 report)
"""


def main():
    raise NotImplementedError("Training loop not written yet")


if __name__ == "__main__":
    main()
