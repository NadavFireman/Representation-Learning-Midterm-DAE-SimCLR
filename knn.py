import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy


@torch.no_grad()
def extract_features(encoder, loader):
    """h for every image in loader, and the labels: ([N, latent_dim], [N]), on the encoder's device.
    From lec5/notebooks/05_moco_experiments.ipynb; the encoder returns to its train/eval mode."""
    was_training = encoder.training
    encoder.eval()
    device = next(encoder.parameters()).device
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):  # mixed precision, as in training
            feats.append(encoder(x.to(device)).float())  # [B, 3, 64, 64] -> [B, latent_dim]
        labels.append(y.to(device))
    encoder.train(was_training)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_scores(train_h, train_y, query_h, num_classes, k=20, chunk=500):
    """Similarity-weighted k-NN vote: one score per class for every query, [N_query, num_classes].

    The assignment's protocol: cosine similarity on L2-normalized h, k = 20, and each
    neighbor adds its similarity to the score of its own class.
    The queries go in chunks, so the [chunk, N_train] similarity matrix stays small.
    """
    train_h, query_h = F.normalize(train_h, dim=1), F.normalize(query_h, dim=1)  # length 1: dot product = cosine
    scores = []
    for i in range(0, len(query_h), chunk):
        sim = query_h[i:i + chunk] @ train_h.T  # [chunk, N_train], cosine similarity
        top_sim, top_idx = sim.topk(k, dim=1)  # the k most similar training images: [chunk, k]
        votes = torch.zeros(len(top_sim), num_classes, device=train_h.device)
        votes.scatter_add_(1, train_y[top_idx], top_sim)  # add each neighbor's similarity to its class
        scores.append(votes)
    return torch.cat(scores)


def knn_accuracy(train_h, train_y, query_h, query_y, num_classes, k=20):
    """k-NN top-1 and top-5 accuracy of the queries (torchmetrics on the vote scores)."""
    scores = knn_scores(train_h, train_y, query_h, num_classes, k)  # [N_query, num_classes]
    top1 = accuracy(scores, query_y, task="multiclass", num_classes=num_classes, top_k=1).item()
    top5 = accuracy(scores, query_y, task="multiclass", num_classes=num_classes, top_k=5).item()
    return top1, top5


def knn_evaluate(encoder, datamodule, k=20, split="val", bank_size=None):
    """k-NN accuracy of a frozen encoder: the training images are the memory bank,
    the val or test images are the queries. bank_size=None uses every training image.
    split can be a tuple, ("val", "test"): one bank, and the result is {split: (top1, top5)}.
    """
    train_h, train_y = extract_features(encoder, datamodule.bank_dataloader(bank_size))
    results = {}
    for name in (split,) if isinstance(split, str) else split:
        query_h, query_y = extract_features(encoder, datamodule.split_dataloader(name, persistent=False))
        results[name] = knn_accuracy(train_h, train_y, query_h, query_y, datamodule.num_classes, k)
    return results[split] if isinstance(split, str) else results
