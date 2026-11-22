import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.models.resnet import BasicBlock

from augmentations import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD

BACKBONES = ("small", "resnet18")  # --backbone: the encoder of every model
DECODERS = ("light", "resnet18")  # --decoder: the decoder of the DAE


def build_encoder(latent_dim, backbone="small"):
    """The encoder f_theta: [B, 3, 64, 64] -> h [B, latent_dim].

    backbone="resnet18" builds the ResNet-18 of build_resnet18_encoder.
    backbone="small", the default, builds the encoder below.

    The small encoder of the lec4 SimCLR class (the assignment's backbone), for
    3x64x64 images instead of 1x28x28. Two stride-2 convolutions halve the image
    twice (64 -> 32 -> 16), so the Linear layer reads 64*16*16 numbers instead of
    64*7*7. The input is already normalized by the transforms, as in the course.
    The course wrote these layers out again in every class. Here one function
    builds them, so every model is sure to get the same encoder, and a saved
    state_dict loads into any of them.
    GELU instead of the course's ReLU: search 2 chose it by val k-NN (0.0697 vs 0.0617).
    """
    if backbone == "resnet18":
        return build_resnet18_encoder(latent_dim)
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, 32, 32, 32]
        nn.GELU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
        nn.GELU(),  # [B, 64, 16, 16] -> [B, 64, 16, 16]
        nn.Flatten(),  # [B, 64, 16, 16] -> [B, 16384]
        nn.Linear(64 * 16 * 16, latent_dim),  # [B, 16384] -> [B, latent_dim]
    )


def build_resnet18_encoder(latent_dim):
    """ResNet-18 as the encoder: [B, 3, 64, 64] -> h [B, latent_dim].

    The ResNet-18 of the course NN class (lec2-lec5): torchvision's model with
    random weights. weights=None matters here: ImageNet weights have already
    seen Tiny-ImageNet's images and labels, since Tiny-ImageNet is cut out of ImageNet.

    The course changed conv1 for 1-channel 28x28 MNIST. For 3-channel 64x64
    images three layers change:
      conv1:   7x7 stride 2 -> 3x3 stride 1. The stock layer shrinks a 224x224
               image by 2 at once; on 64x64 that throws away detail.
      maxpool: removed (Identity), for the same reason. With both stock layers
               the last feature map would be 2x2; now it is 8x8.
      fc:      the stock 1000-way layer -> Linear(512, latent_dim), so h has the
               same length as with the small encoder, and the decoder, the k-NN
               and the heads work unchanged.
    """
    encoder = torchvision.models.resnet18(weights=None, num_classes=latent_dim)  # fc: [B, 512] -> [B, latent_dim]
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # [B, 3, 64, 64] -> [B, 64, 64, 64]
    encoder.maxpool = nn.Identity()  # [B, 64, 64, 64] -> [B, 64, 64, 64]
    # layer1: [B, 64, 64, 64], layer2: [B, 128, 32, 32], layer3: [B, 256, 16, 16],
    # layer4: [B, 512, 8, 8], avgpool + flatten: [B, 512], fc: [B, latent_dim]
    return encoder


def build_decoder(latent_dim, decoder="light"):
    """The decoder g_phi: h [B, latent_dim] -> [B, 3, 64, 64].

    decoder="light", the default: the mirror of the small encoder (16 -> 32 -> 64),
    as in the course AutoEncoder. As there, the last layer has no activation: the
    clean target is normalized, so its pixels run from about -1.7 to 2.1, and a
    Sigmoid, (0, 1), could never match them.
    decoder="resnet18": build_resnet18_decoder.
    """
    if decoder == "resnet18":
        return build_resnet18_decoder(latent_dim)
    return nn.Sequential(
        nn.Linear(latent_dim, 64 * 16 * 16),  # [B, latent_dim] -> [B, 16384]
        nn.GELU(),  # [B, 16384] -> [B, 16384]
        nn.Unflatten(1, (64, 16, 16)),  # [B, 16384] -> [B, 64, 16, 16]
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 16, 16] -> [B, 32, 32, 32]
        nn.GELU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 32, 32] -> [B, 3, 64, 64], same scale as the normalized input
    )


def resnet18_up_stage(in_channels, out_channels, upsample):
    """One stage of the inverted ResNet-18: two BasicBlocks, then double H and W.

    A ResNet-18 stage halves H and W in its first block and keeps them in the
    second. This stage runs the other way. The BasicBlocks are torchvision's own,
    the ones inside the encoder. When the number of channels changes, BasicBlock
    needs a 1x1 convolution on its shortcut, as ResNet itself uses.
    """
    shortcut = None
    if in_channels != out_channels:
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # [B, in, H, W] -> [B, out, H, W]
            nn.BatchNorm2d(out_channels),
        )
    layers = [
        BasicBlock(in_channels, out_channels, downsample=shortcut),  # [B, in, H, W] -> [B, out, H, W]
        BasicBlock(out_channels, out_channels),  # [B, out, H, W] -> [B, out, H, W]
    ]
    if upsample:
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))  # [B, out, H, W] -> [B, out, 2H, 2W]
    return nn.Sequential(*layers)


def build_resnet18_decoder(latent_dim):
    """The inverted ResNet-18: h [B, latent_dim] -> [B, 3, 64, 64].

    Lecture 3 [00:36:37-00:37:16, 00:43:43-00:45:08]: a decoder that is ResNet-18
    backwards, layer 4, 3, 2, 1, written by us with the BasicBlock, since
    torchvision has no inverted ResNet. Each stage mirrors one of
    build_resnet18_encoder, so the shapes run in reverse:
        fc^-1      Linear + reshape     [B, latent_dim] -> [B, 512, 8, 8]
        layer4^-1  512 -> 256, x2       -> [B, 256, 16, 16]
        layer3^-1  256 -> 128, x2       -> [B, 128, 32, 32]
        layer2^-1  128 -> 64,  x2       -> [B, 64, 64, 64]
        layer1^-1  64 -> 64             -> [B, 64, 64, 64]
        conv1^-1   3x3 conv to 3 colors -> [B, 3, 64, 64]
    No skip connections from the encoder (that would be a U-Net): everything
    still passes through h, or the encoder would not need to learn anything.
    This decoder is much stronger than the light one. Lecture 3 [00:41:57]
    asks whether a weaker decoder makes the encoder learn more; running both
    decoders with the same encoder tests that.
    """
    return nn.Sequential(
        nn.Linear(latent_dim, 512 * 8 * 8),  # [B, latent_dim] -> [B, 32768]
        nn.ReLU(),  # [B, 32768] -> [B, 32768]
        nn.Unflatten(1, (512, 8, 8)),  # [B, 32768] -> [B, 512, 8, 8]
        resnet18_up_stage(512, 256, upsample=True),  # [B, 512, 8, 8] -> [B, 256, 16, 16]
        resnet18_up_stage(256, 128, upsample=True),  # [B, 256, 16, 16] -> [B, 128, 32, 32]
        resnet18_up_stage(128, 64, upsample=True),  # [B, 128, 32, 32] -> [B, 64, 64, 64]
        resnet18_up_stage(64, 64, upsample=False),  # [B, 64, 64, 64] -> [B, 64, 64, 64]
        nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 64, 64, 64] -> [B, 3, 64, 64], no activation (normalized target)
    )


class DenoisingAutoEncoder(pl.LightningModule):
    """Denoising autoencoder: corrupted image in, clean image out.

    The course AutoEncoder (lec5) with one change in the loss. The input is corrupted,
    and the loss is MSE between the reconstruction and the clean image. To undo
    the corruption, the encoder has to keep what the image shows and drop the
    noise, and that is the representation h we want. No skip connections (not
    a U-Net): everything passes through h.

    The images come normalized from the transforms, as in the course, so the noise
    works on normalized pixels. noise_type, the three corruptions the assignment
    lists for a DAE:
        "gaussian":    x + noise_level * N(0, 1), in units of the per-channel std
        "mask":        a noise_level fraction of the pixels is set to 0, which after
                       normalization is the mean color (all 3 channels)
        "salt_pepper": a noise_level fraction of the pixels is set to black or white, half each
    noise_level=0 changes nothing, which is the plain autoencoder of the course.
    That is the AE baseline, with no separate class.

    The noise is added here and not in the transforms, because the loss needs
    the clean image as well. It is added in val and test too, so val_loss
    measures the same task the model trains on. In val and test the noise of
    batch i always comes from seed i, so every epoch and every trial is scored
    on the same corrupted images and their reconstruction errors can be compared.

    backbone and decoder pick the two halves (build_encoder, build_decoder). The
    defaults, "small" and "light", are the course encoder and its mirror.
    After pretraining only the encoder is kept.

    The optimizer is AdamW with weight_decay, chosen by search 2 over Adam and RAdam.
    """

    def __init__(self, learning_rate, latent_dim, noise_type, noise_level, weight_decay=0.0,
                 backbone="small", decoder="light"):
        super().__init__()
        self.save_hyperparameters()  # learning_rate, latent_dim, noise_type, noise_level, weight_decay, backbone, decoder -> self.hparams
        # Black (pixel 0) and white (pixel 1) after normalization, per channel, for salt_pepper.
        # Buffers: they move to the GPU with the model, and the optimizer never changes them.
        mean = torch.tensor(TINY_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(TINY_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("black", (0 - mean) / std)  # [1, 3, 1, 1]
        self.register_buffer("white", (1 - mean) / std)

        self.encoder = build_encoder(latent_dim, backbone)
        self.decoder = build_decoder(latent_dim, decoder)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.encoder(x)     # [B, 3, 64, 64] -> [B, latent_dim]
        return self.decoder(h)  # [B, latent_dim] -> [B, 3, 64, 64]

    def add_noise(self, x, generator=None):
        """Corrupt the normalized images x [B, 3, 64, 64].

        generator=None draws new noise every call (training). A seeded generator
        gives the same noise every time (val and test).
        """
        noise_type, noise_level = self.hparams.noise_type, self.hparams.noise_level
        if noise_type == "gaussian":
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            return x + noise_level * noise
        # mask and salt_pepper pick whole pixels: one random number per pixel, shared by the 3 channels
        r = torch.rand((x.size(0), 1, x.size(2), x.size(3)), generator=generator, device=x.device, dtype=x.dtype)  # [B, 1, 64, 64]
        if noise_type == "mask":
            return x * (r >= noise_level)  # a noise_level fraction of the pixels -> 0, the mean color
        if noise_type == "salt_pepper":
            x = torch.where(r < noise_level / 2, self.black, x)  # pepper: -> black
            return torch.where(r > 1 - noise_level / 2, self.white, x)  # salt: -> white
        raise ValueError(f"unknown noise_type: {noise_type}")

    def _fixed_generator(self, batch_idx):
        """The same noise for batch batch_idx in every epoch and every trial."""
        return torch.Generator(device=self.device).manual_seed(batch_idx)

    def training_step(self, batch, batch_idx):
        loss, recon, x, x_noisy = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._log_reconstructions(x, x_noisy, recon, "train_reconstructions")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, x, x_noisy = self._common_step(batch, self._fixed_generator(batch_idx))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0 and not self.trainer.sanity_checking:
            self._log_reconstructions(x, x_noisy, recon, "val_reconstructions")
        return loss

    def test_step(self, batch, batch_idx):
        loss, recon, x, x_noisy = self._common_step(batch, self._fixed_generator(batch_idx))
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._log_reconstructions(x, x_noisy, recon, "test_reconstructions")
        return loss

    def _common_step(self, batch, generator=None):
        x, _ = batch  # x: [B, 3, 64, 64], clean; the label is unused
        x_noisy = self.add_noise(x, generator)
        recon = self(x_noisy)
        loss = self.loss_fn(recon, x)  # compared with the clean image, not the noisy one
        return loss, recon, x, x_noisy

    def _log_reconstructions(self, x, x_noisy, recon, tag):
        """Top row: clean inputs. Middle row: noisy inputs. Bottom row: reconstructions."""
        if self.logger is None:
            return
        n = min(8, x.size(0))
        images = torch.cat([x[:n], x_noisy[:n], recon[:n]]).detach().float().cpu()
        grid = torchvision.utils.make_grid(images, nrow=n, normalize=True)  # normalize=True: rescaled to [0, 1] for display
        self.logger.experiment.add_image(tag, grid, self.global_step)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)


class EncoderClassifier(pl.LightningModule):
    """Encoder plus a new linear head. Used for linear probe, fine-tuning and transfer.

    encoder_path is the state_dict saved after pretraining. encoder_path=None
    keeps the random initial weights, which is the random-init baseline.
    freeze_encoder=True trains only the head (linear probe).
    False trains the encoder and the head (fine-tuning).

    Logs top-1 and top-5 accuracy, as the assignment asks, and on val and test also
    macro precision, recall, F1 and AUROC; on test also the confusion matrix.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False,
                 backbone="small", label_smoothing=0.0):
        super().__init__()
        self.save_hyperparameters()  # learning_rate, freeze_encoder, backbone, ... -> self.hparams

        # backbone must be the one the saved encoder was built with, or its state_dict won't load.
        self.encoder = build_encoder(latent_dim, backbone)
        if encoder_path is not None:
            # Weights from pretraining. The head below is not in this file.
            state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict)

        self.head = nn.Linear(latent_dim, num_classes)  # [B, latent_dim] -> [B, num_classes]

        # eval() before freezing: a frozen encoder must not update BatchNorm stats
        # or dropout. train() below keeps that, because Lightning calls train()
        # at the start of every epoch and would otherwise undo it.
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        # label_smoothing > 0 (config.PROTOCOL "acc"): the target puts 1 - label_smoothing on
        # the true class, so the model is not pushed to be 100% sure, and val_loss
        # does not climb from overconfidence while val_accuracy still improves.
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # The metrics. Top-1 and top-5 accuracy are the ones the assignment asks for.
        # Precision, recall and F1 are macro: one value per class, then their mean, so
        # every class counts the same. (Val and test have 50 images per class, so the
        # micro versions would all equal accuracy, and macro recall equals it too.)
        # AUROC: for each class, how well the scores rank its images above the others
        # (one-vs-rest), then the mean over the classes.
        # Each stage gets its own copy, so train, val and test never mix their counts.
        metrics = torchmetrics.MetricCollection({
            # average="micro": correct images / all images, as torchmetrics.Accuracy computed before
            # (the Multiclass classes default to macro).
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes, average="micro"),
            "top5_accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5, average="micro"),
            "precision": torchmetrics.classification.MulticlassPrecision(num_classes, average="macro"),
            "recall": torchmetrics.classification.MulticlassRecall(num_classes, average="macro"),
            "f1": torchmetrics.classification.MulticlassF1Score(num_classes, average="macro"),
            "auroc": torchmetrics.classification.MulticlassAUROC(num_classes, average="macro"),
        })
        self.val_metrics = metrics.clone(prefix="val_")  # val_accuracy, val_f1, ...: val_accuracy drives EarlyStopping
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes, average="micro")
        # Rows: the true class, columns: the predicted class. callbacks.ConfusionMatrixCallback saves it after the test.
        self.test_confusion = torchmetrics.classification.MulticlassConfusionMatrix(num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.hparams.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        h = self.encoder(x)  # [B, 3, 64, 64] -> [B, latent_dim]
        return self.head(h)  # [B, latent_dim] -> [B, num_classes]

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.train_accuracy(scores, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": self.train_accuracy},
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        # update() only adds this batch to the counts. Logging the metric objects lets
        # Lightning compute each one once, over the whole val set, at the end of the epoch.
        # float(): with 16-mixed the scores are float16, and AUROC's softmax over 200
        # classes would round many small probabilities to the same value.
        self.val_metrics.update(scores.float(), y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.test_metrics.update(scores.float(), y)  # float(): see validation_step
        self.test_confusion.update(scores, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, 64, 64]
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def configure_optimizers(self):
        # Linear probing: only the head is trainable, so only the head is optimized.
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.hparams.learning_rate)


class SupervisedEncoder(EncoderClassifier):
    """The same encoder and head, trained from scratch with labels (supervised baseline).

    The course wrote this class out in full. It is exactly EncoderClassifier with
    no loaded weights and nothing frozen, so here it is that, under its own name.

    EncoderClassifier saves encoder_path=None and freeze_encoder=False in the
    hyperparameters, and load_from_checkpoint passes them back to this __init__.
    **ignored takes them, so a saved SupervisedEncoder can be loaded again;
    the values are always None and False here.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, backbone="small", label_smoothing=0.0, **ignored):
        super().__init__(learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False,
                         backbone=backbone, label_smoothing=label_smoothing)
