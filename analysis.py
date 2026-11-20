"""--mode analysis: the evidence of A5 and the robustness test of A4 for one frozen encoder.

No training. Everything is computed on the test set with the k-NN of the
evaluation protocol (k=20, cosine, similarity-weighted vote, the training images
as the memory bank), so the numbers match the knn mode:
    per_class_accuracy.csv   k-NN accuracy of every class: where the representation fails
    tsne.png                 t-SNE of h for 10 classes of the test set
    retrieval.png            8 test images and their 8 nearest training images
    robustness.csv / .png    k-NN accuracy when the test images are corrupted
The figures also go to the run's TensorBoard (IMAGES tab).
"""
import os

import matplotlib
matplotlib.use("Agg")  # no screen on Colab: draw into files
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchmetrics.functional import accuracy
from torchvision.transforms.v2 import functional as TF

import config
from augmentations import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
from knn import extract_features, knn_scores

# The robustness test (A4 bonus): the bank stays clean, only the test images are
# corrupted, with corruptions the DAE did not train on (it trained on salt_pepper).
# Each at 3 severities, from mild to strong. Gaussian noise is in units of the
# per-channel std, as the DAE's noise; blur is the sigma of the kernel in pixels;
# occlusion is the side of a gray square in the middle of the image, as a fraction of 64.
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
        feats.append(encoder(corrupt(x.to(device), name, severity, generator)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def class_names(dm):
    """Tiny-ImageNet: the WordNet ids (n01443537 ...). CIFAR-100: the class names."""
    if config.DATASET == "TinyImageNet":
        return dm.test_ds.hf_split.features["label"].names
    return dm.test_ds.classes


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
    names = class_names(dm)
    k, num_classes = config.KNN_K, dm.num_classes

    bank_h, bank_y = extract_features(encoder, dm.bank_dataloader())
    test_h, test_y = extract_features(encoder, dm.test_dataloader())
    scores = knn_scores(bank_h, bank_y, test_h, num_classes, k)  # [N_test, num_classes]
    results = {
        "test_knn_top1": accuracy(scores, test_y, task="multiclass", num_classes=num_classes).item(),
        "test_knn_top5": accuracy(scores, test_y, task="multiclass", num_classes=num_classes, top_k=5).item(),
    }

    # 1. Per-class k-NN accuracy: which classes the representation fails on.
    per_class = accuracy(scores, test_y, task="multiclass", num_classes=num_classes, average=None)
    per_class = pd.DataFrame({"class": names, "accuracy": per_class.numpy()}).sort_values("accuracy")
    per_class.to_csv(os.path.join(logger.log_dir, "per_class_accuracy.csv"), index=False)
    print("Worst classes:\n", per_class.head(10).to_string(index=False))
    print("Best classes:\n", per_class.tail(10).to_string(index=False))

    # 2. t-SNE of h (L2-normalized, as the k-NN sees it) for the first n_tsne_classes classes.
    keep = test_y < n_tsne_classes
    points = TSNE(n_components=2, init="pca", random_state=config.SEED).fit_transform(
        F.normalize(test_h[keep], dim=1).numpy()
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=test_y[keep].numpy(), cmap="tab10", s=10)
    ax.legend(scatter.legend_elements()[0], names[:n_tsne_classes], fontsize=7, loc="best")
    ax.set_title(f"t-SNE of h, {n_tsne_classes} test classes of {config.DATASET}")
    save_figure(fig, logger, "tsne")

    # 3. Retrieval: test queries (one from each of n_queries classes) and their nearest training images.
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
    robustness.to_csv(os.path.join(logger.log_dir, "robustness.csv"), index=False)
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
    logger.save()  # writes hparams.yaml (no Trainer here to do it)
    return results
