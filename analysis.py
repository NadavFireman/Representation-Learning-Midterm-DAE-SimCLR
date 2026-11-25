"""--mode analysis: the evidence of A5 and the robustness test of A4 for one frozen encoder.

No training. The k-NN of the evaluation protocol on the test set gives:
    per_class_accuracy.csv   k-NN accuracy of every class
    knn_confusion_matrix.*   the k-NN confusion matrix, and knn_most_confused.csv
    tsne.png                 t-SNE of h for 10 classes of the test set
    retrieval.png            8 test images and their 8 nearest training images
    robustness.csv / .png    k-NN accuracy when the test images are corrupted
"""
import os

import matplotlib
matplotlib.use("Agg")  # no screen on Colab: draw into files
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score, precision, recall
from torchvision.transforms.v2 import functional as TF

import config
from augmentations import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
from knn import extract_features, knn_scores

# The robustness test (A4 bonus): a clean bank, test images corrupted at 3 severities.
# gaussian_noise: std; blur: sigma in pixels; occlusion: side of a gray square, as a fraction of 64.
CORRUPTIONS = {
    "gaussian_noise": (0.25, 0.5, 1.0),
    "blur": (0.5, 1.0, 2.0),
    "occlusion": (0.25, 0.4, 0.55),
}


def corrupt(x, name, severity, generator):
    """Corrupt a batch of normalized images x [B, 3, 64, 64]."""
    if name == "gaussian_noise":
        return x + severity * torch.randn(x.shape, generator=generator, device=x.device)
    if name == "blur":
        kernel = 2 * round(3 * severity) + 1  # odd, about 3 sigmas on each side
        return TF.gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[severity, severity])
    if name == "occlusion":
        side = round(severity * x.size(-1))
        start = (x.size(-1) - side) // 2
        x = x.clone()
        x[..., start:start + side, start:start + side] = 0  # 0 after normalization = the mean color
        return x
    raise ValueError(f"unknown corruption: {name}")


@torch.no_grad()
def extract_corrupted(encoder, loader, name, severity):
    """h of the corrupted test images. A fixed seed: every encoder sees the same noise."""
    device = next(encoder.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.SEED)
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            feats.append(encoder(corrupt(x.to(device), name, severity, generator)).float())
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


def knn_metrics(scores, y, num_classes, prefix="test_knn_"):
    """Macro precision, recall, F1 and AUROC of the k-NN votes.
    For AUROC the scores become each class's share of the votes, so every row sums to 1."""
    shares = scores.clamp(min=0)
    shares = shares / shares.sum(dim=1, keepdim=True).clamp(min=1e-12)
    args = dict(task="multiclass", num_classes=num_classes, average="macro")
    return {
        f"{prefix}precision": precision(scores, y, **args).item(),
        f"{prefix}recall": recall(scores, y, **args).item(),
        f"{prefix}f1": f1_score(scores, y, **args).item(),
        f"{prefix}auroc": auroc(shares, y, **args).item(),
    }


def log_table(logger, name, table):
    """A pandas table to a CSV in the run folder and to TensorBoard's TEXT tab."""
    table.to_csv(os.path.join(logger.log_dir, f"{name}.csv"), index=False)
    rows = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    rows += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    logger.experiment.add_text(name, "\n".join(rows))


def save_confusion_matrix(confusion, names, logger, prefix="", n_pairs=20):
    """Save a confusion matrix [num_classes, num_classes] (rows: true, columns: predicted):
        <prefix>confusion_matrix.csv  the counts, with the class names
        <prefix>confusion_matrix.png  each row divided by its class size
        <prefix>most_confused.csv     the n_pairs largest off-diagonal cells
    """
    confusion = confusion.cpu().numpy()
    pd.DataFrame(confusion, index=names, columns=names).to_csv(
        os.path.join(logger.log_dir, f"{prefix}confusion_matrix.csv")
    )
    class_size = confusion.sum(axis=1, keepdims=True).clip(min=1)
    mistakes = confusion.copy()
    np.fill_diagonal(mistakes, 0)  # only the mistakes
    largest = np.argsort(mistakes, axis=None)[::-1][:n_pairs]  # flat indices, largest first
    rows = [
        {"true": names[i], "predicted": names[j], "count": int(mistakes[i, j]),
         "share_of_true_class": mistakes[i, j] / class_size[i, 0]}
        for i, j in zip(*np.unravel_index(largest, mistakes.shape))
        if mistakes[i, j] > 0  # skip pairs that were never confused
    ]
    log_table(logger, f"{prefix}most_confused", pd.DataFrame(rows, columns=["true", "predicted", "count", "share_of_true_class"]))

    fig, ax = plt.subplots(figsize=(7, 6))
    # PowerNorm (gamma 0.3) spreads the small shares out; on a linear scale they are one color.
    image = ax.imshow(confusion / class_size, cmap="viridis", norm=PowerNorm(gamma=0.3, vmin=0, vmax=1))
    fig.colorbar(image, ax=ax, label="share of the true class")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Confusion matrix on test ({len(names)} classes)")
    save_figure(fig, logger, f"{prefix}confusion_matrix")


def to_image(x):
    """A normalized image [3, 64, 64] back to pixels in [0, 1] [64, 64, 3], for display."""
    mean = torch.tensor(TINY_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(TINY_IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_figure(fig, logger, name):
    fig.savefig(os.path.join(logger.log_dir, f"{name}.png"), dpi=120, bbox_inches="tight")
    logger.experiment.add_figure(name, fig)
    plt.close(fig)


def run_analysis(encoder, dm, logger, n_tsne_classes=10, n_queries=8, n_neighbors=8):
    dm.prepare_data()
    dm.setup("test")
    encoder.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    os.makedirs(logger.log_dir, exist_ok=True)
    names = dm.class_names()
    k, num_classes = config.KNN_K, dm.num_classes

    bank_h, bank_y = extract_features(encoder, dm.bank_dataloader())
    test_h, test_y = extract_features(encoder, dm.test_dataloader())
    scores = knn_scores(bank_h, bank_y, test_h, num_classes, k)  # [N_test, num_classes]
    results = {
        "test_knn_top1": accuracy(scores, test_y, task="multiclass", num_classes=num_classes).item(),
        "test_knn_top5": accuracy(scores, test_y, task="multiclass", num_classes=num_classes, top_k=5).item(),
    }
    results.update(knn_metrics(scores, test_y, num_classes))

    # 0. The confusion matrix of the k-NN.
    save_confusion_matrix(confusion_matrix(scores, test_y, task="multiclass", num_classes=num_classes),
                          names, logger, prefix="knn_")

    # 1. Per-class k-NN accuracy.
    per_class = accuracy(scores, test_y, task="multiclass", num_classes=num_classes, average=None)
    per_class = pd.DataFrame({"class": names, "accuracy": per_class.cpu().numpy()}).sort_values("accuracy")
    log_table(logger, "per_class_accuracy", per_class)
    print("Worst classes:\n", per_class.head(10).to_string(index=False))
    print("Best classes:\n", per_class.tail(10).to_string(index=False))

    # 2. t-SNE of the L2-normalized h, for the first n_tsne_classes classes.
    keep = test_y < n_tsne_classes
    points = TSNE(n_components=2, init="pca", random_state=config.SEED).fit_transform(
        F.normalize(test_h[keep], dim=1).cpu().numpy()
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=test_y[keep].cpu().numpy(), cmap="tab10", s=10)
    ax.legend(scatter.legend_elements()[0], names[:n_tsne_classes], fontsize=7, loc="best")
    ax.set_title(f"t-SNE of h, {n_tsne_classes} test classes of {config.DATASET}")
    save_figure(fig, logger, "tsne")

    # 3. Retrieval: test queries and their nearest training images.
    queries = [int((test_y == c).nonzero()[0]) for c in range(0, num_classes, num_classes // n_queries)][:n_queries]
    sim = F.normalize(test_h[queries], dim=1) @ F.normalize(bank_h, dim=1).T  # [n_queries, N_bank]
    nearest = sim.topk(n_neighbors, dim=1).indices
    fig, axes = plt.subplots(n_queries, n_neighbors + 1, figsize=(1.3 * (n_neighbors + 1), 1.4 * n_queries))
    for row, q in enumerate(queries):
        query_x, query_y = dm.test_ds[q]
        axes[row, 0].imshow(to_image(query_x))
        axes[row, 0].set_title("query", fontsize=7)
        for col, i in enumerate(nearest[row].tolist(), start=1):
            x, y = dm.bank_ds[i]
            axes[row, col].imshow(to_image(x))
            for spine in axes[row, col].spines.values():  # green frame: same class as the query
                spine.set(color="green" if y == query_y else "red", linewidth=2)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Nearest training images (green: same class)", fontsize=9)
    save_figure(fig, logger, "retrieval")

    # 4. Robustness: k-NN top-1 on corrupted test images, against the clean bank.
    rows = [{"corruption": "none", "severity": 0, "test_knn_top1": results["test_knn_top1"]}]
    for name, severities in CORRUPTIONS.items():
        for severity in severities:
            h, y = extract_corrupted(encoder, dm.test_dataloader(), name, severity)
            top1 = accuracy(knn_scores(bank_h, bank_y, h, num_classes, k), y,
                            task="multiclass", num_classes=num_classes).item()
            rows.append({"corruption": name, "severity": severity, "test_knn_top1": top1})
            results[f"robust_{name}_{severity}"] = top1
    robustness = pd.DataFrame(rows)
    log_table(logger, "robustness", robustness)
    print(robustness.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, severities in CORRUPTIONS.items():
        part = robustness[robustness.corruption == name]
        ax.plot(range(len(severities) + 1), [results["test_knn_top1"], *part.test_knn_top1], marker="o", label=name)
    ax.set_xticks(range(4), ["clean", "mild", "medium", "strong"])
    ax.set_ylabel("test k-NN top-1")
    ax.legend()
    save_figure(fig, logger, "robustness")

    logger.log_metrics(results)
    logger.save()  # writes hparams.yaml
    return results
