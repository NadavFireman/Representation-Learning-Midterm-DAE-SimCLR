import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy


@torch.no_grad()
def extract_features(encoder, loader):
    """h for every image in loader, and the labels: ([N, latent_dim], [N]).

    From extract_features in lec5/notebooks/05_moco_experiments.ipynb. Two changes:
    the device is the encoder's own (so it also works inside training), and the
    encoder goes back to train mode afterwards if it was training.
    """
    was_training = encoder.training
    encoder.eval()
    device = next(encoder.parameters()).device
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device)).float().cpu())  # [B, 3, 64, 64] -> [B, latent_dim]
        labels.append(y)
    encoder.train(was_training)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_scores(train_h, train_y, query_h, num_classes, k=20, chunk=500):
    """Similarity-weighted k-NN vote: one score per class for every query, [N_query, num_classes].

    From knn_accuracy in the same notebook, with the assignment's protocol:
      - cosine similarity on L2-normalized h (as in the notebook)
      - k = 20 (the notebook used 10)
      - similarity-weighted vote (the notebook used a plain majority, .mode()):
        each of the k neighbors adds its cosine similarity to the score of its
        own class, so a close neighbor counts more than a far one.
    Only torch built-ins: F.normalize, a matrix product, topk and scatter_add_.
    scikit-learn's KNeighborsClassifier gives the same scores, but with
    metric="cosine" it can't use its search trees and sorts each row in numpy,
    about 8x slower on 90,000 x 10,000. The queries go in chunks of 500, so the
    [chunk, N_train] similarity matrix stays small (500 x 90,000 floats = 180 MB).
    """
    train_h, query_h = F.normalize(train_h, dim=1), F.normalize(query_h, dim=1)  # length 1: dot product = cosine
    scores = []
    for i in range(0, len(query_h), chunk):
        sim = query_h[i:i + chunk] @ train_h.T  # [chunk, N_train], cosine similarity
        top_sim, top_idx = sim.topk(k, dim=1)  # the k most similar training images: [chunk, k]
        votes = torch.zeros(len(top_sim), num_classes)
        votes.scatter_add_(1, train_y[top_idx], top_sim)  # add each neighbor's similarity to its class
        scores.append(votes)
    return torch.cat(scores)


def knn_accuracy(train_h, train_y, query_h, query_y, num_classes, k=20):
    """k-NN top-1 and top-5 accuracy of the queries. No training is involved.

    torchmetrics turns the vote scores into top-1 and top-5 accuracy, as in EncoderClassifier.
    """
    scores = knn_scores(train_h, train_y, query_h, num_classes, k)  # [N_query, num_classes]
    top1 = accuracy(scores, query_y, task="multiclass", num_classes=num_classes, top_k=1).item()
    top5 = accuracy(scores, query_y, task="multiclass", num_classes=num_classes, top_k=5).item()
    return top1, top5


def knn_evaluate(encoder, datamodule, k=20, split="val", bank_size=None):
    """k-NN accuracy of a frozen encoder: the training images are the memory bank,
    the val (model selection) or test (final numbers) images are the queries.

    The bank comes from bank_dataloader: every training image whatever the label
    fraction, no flip and a fixed order, so every model is scored with the same
    bank and the same encoder always gets the same accuracy.
    bank_size=None uses every training image.
    """
    train_h, train_y = extract_features(encoder, datamodule.bank_dataloader(bank_size))
    loader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()
    query_h, query_y = extract_features(encoder, loader)
    return knn_accuracy(train_h, train_y, query_h, query_y, datamodule.num_classes, k)
