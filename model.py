import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchvision.models.resnet import BasicBlock

from augmentations import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD

BACKBONES = ("small", "resnet18")  # --backbone: the encoder of every model
DECODERS = ("light", "resnet18")  # --decoder: the decoder of the DAE
# --activation of the encoder; None: the backbone's own (CNN-3: GELU, chosen by the search; ResNet-18: ReLU)
ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "silu": nn.SiLU, "mish": nn.Mish}
# --optimizer of the classifiers. SGD with Nesterov momentum; the others are Adam variants.
OPTIMIZERS = {"adamw": optim.AdamW, "adam": optim.Adam, "nadam": optim.NAdam, "radam": optim.RAdam,
              "sgd": lambda groups, **kw: optim.SGD(groups, momentum=0.9, nesterov=True, **kw)}
# --noise_type of the DAE; patch_mask (MAE) and block (context encoder) hide regions, not pixels
NOISE_TYPES = ("gaussian", "mask", "salt_pepper", "patch_mask", "block")
SCHEDULERS = ("none", "cosine")  # --scheduler of the DAE


def build_encoder(latent_dim, backbone="small", activation=None, channels=(32, 64)):
    """The encoder f_theta: [B, 3, 64, 64] -> h [B, latent_dim].

    "small": the encoder of the lec4 SimCLR class (the assignment's backbone), for 3x64x64
    images. GELU instead of ReLU: chosen by the search. channels: the output channels of its
    two conv layers. "resnet18": build_resnet18_encoder.
    activation: one of ACTIVATIONS in place of the backbone's own.
    """
    if backbone == "resnet18":
        encoder = build_resnet18_encoder(latent_dim)
        if activation is not None:
            swap_activation(encoder, ACTIVATIONS[activation])
        return encoder
    act = ACTIVATIONS[activation or "gelu"]
    c1, c2 = channels
    return nn.Sequential(
        nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, c1, 32, 32]
        act(),  # [B, c1, 32, 32] -> [B, c1, 32, 32]
        nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),  # [B, c1, 32, 32] -> [B, c2, 16, 16]
        act(),  # [B, c2, 16, 16] -> [B, c2, 16, 16]
        nn.Flatten(),  # [B, c2, 16, 16] -> [B, c2 * 256]
        nn.Linear(c2 * 16 * 16, latent_dim),  # [B, c2 * 256] -> [B, latent_dim]
    )


def swap_activation(module, act):
    """Replace every nn.ReLU inside module (torchvision's ResNet) with act()."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, act())
        else:
            swap_activation(child, act)


def build_resnet18_encoder(latent_dim):
    """ResNet-18 as the encoder: [B, 3, 64, 64] -> h [B, latent_dim].

    torchvision's model with random weights (ImageNet weights have seen Tiny-ImageNet).
    For 64x64 images: conv1 is 3x3 stride 1 and maxpool is removed, so the last feature
    map is 8x8 and not 2x2; fc gives h, the same length as with the small encoder.
    """
    encoder = torchvision.models.resnet18(weights=None, num_classes=latent_dim)  # fc: [B, 512] -> [B, latent_dim]
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # [B, 3, 64, 64] -> [B, 64, 64, 64]
    encoder.maxpool = nn.Identity()  # [B, 64, 64, 64] -> [B, 64, 64, 64]
    # layer1: [B, 64, 64, 64], layer2: [B, 128, 32, 32], layer3: [B, 256, 16, 16],
    # layer4: [B, 512, 8, 8], avgpool + flatten: [B, 512], fc: [B, latent_dim]
    return encoder


def build_decoder(latent_dim, decoder="light", channels=(32, 64), activation=None):
    """The decoder g_phi: h [B, latent_dim] -> [B, 3, 64, 64].

    "light": the mirror of the small encoder (same channels and activation), as in the course
    AutoEncoder. No activation on the last layer: the target is normalized. "resnet18": build_resnet18_decoder.
    """
    if decoder == "resnet18":
        return build_resnet18_decoder(latent_dim)
    act = ACTIVATIONS[activation or "gelu"]
    c1, c2 = channels
    return nn.Sequential(
        nn.Linear(latent_dim, c2 * 16 * 16),  # [B, latent_dim] -> [B, c2 * 256]
        act(),  # [B, c2 * 256] -> [B, c2 * 256]
        nn.Unflatten(1, (c2, 16, 16)),  # [B, c2 * 256] -> [B, c2, 16, 16]
        nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, c2, 16, 16] -> [B, c1, 32, 32]
        act(),  # [B, c1, 32, 32] -> [B, c1, 32, 32]
        nn.ConvTranspose2d(c1, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, c1, 32, 32] -> [B, 3, 64, 64]
    )


def resnet18_up_stage(in_channels, out_channels, upsample):
    """One stage of the inverted ResNet-18: two torchvision BasicBlocks, then double H and W.
    When the channels change, the shortcut is a 1x1 convolution, as in ResNet."""
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
    """The inverted ResNet-18 (lecture 3): h [B, latent_dim] -> [B, 3, 64, 64].

    Each stage mirrors one stage of build_resnet18_encoder, in reverse order.
    No skip connections from the encoder: everything passes through h.
    """
    return nn.Sequential(
        nn.Linear(latent_dim, 512 * 8 * 8),  # [B, latent_dim] -> [B, 32768]
        nn.ReLU(),  # [B, 32768] -> [B, 32768]
        nn.Unflatten(1, (512, 8, 8)),  # [B, 32768] -> [B, 512, 8, 8]
        resnet18_up_stage(512, 256, upsample=True),  # [B, 512, 8, 8] -> [B, 256, 16, 16]
        resnet18_up_stage(256, 128, upsample=True),  # [B, 256, 16, 16] -> [B, 128, 32, 32]
        resnet18_up_stage(128, 64, upsample=True),  # [B, 128, 32, 32] -> [B, 64, 64, 64]
        resnet18_up_stage(64, 64, upsample=False),  # [B, 64, 64, 64] -> [B, 64, 64, 64]
        nn.Conv2d(64, 3, kernel_size=3, padding=1),  # [B, 64, 64, 64] -> [B, 3, 64, 64], no activation
    )


class DenoisingAutoEncoder(pl.LightningModule):
    """Denoising autoencoder: corrupted image in, clean image out.

    The course AutoEncoder (lec5) with a corrupted input: the loss is MSE between the
    reconstruction and the clean image. noise_type:
        "gaussian":    x + noise_level * N(0, 1)
        "mask":        a noise_level fraction of the pixels is set to 0 (the mean color)
        "salt_pepper": a noise_level fraction of the pixels is set to black or white
        "patch_mask":  a noise_level fraction of the patch_size x patch_size patches is set to 0 (MAE)
        "block":       one random square, noise_level of the image, is set to 0 (context encoder)
    noise_level=0 is the plain autoencoder (the AE baseline).
    noise_type2, noise_level2: a second noise added on top of the first ("none": no second noise).
    In val and test the noise of batch i comes from seed i, so every epoch is scored
    on the same corrupted images.
    channels, activation: of the encoder (build_encoder); the light decoder mirrors them.
    dropout: share of h zeroed before the decoder, in training only. optimizer: OPTIMIZERS;
    scheduler "cosine": the lr decays to 0 over the run (CosineAnnealingLR).
    """

    def __init__(self, learning_rate, latent_dim, noise_type, noise_level, weight_decay=0.0,
                 backbone="small", decoder="light", activation=None, channels=(32, 64), dropout=0.0,
                 optimizer="adamw", scheduler="none", patch_size=8,
                 noise_type2="none", noise_level2=0.0):
        super().__init__()
        self.save_hyperparameters()  # the arguments -> self.hparams
        # Black (pixel 0) and white (pixel 1) after normalization, for salt_pepper
        mean = torch.tensor(TINY_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(TINY_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("black", (0 - mean) / std)  # [1, 3, 1, 1]
        self.register_buffer("white", (1 - mean) / std)

        self.encoder = build_encoder(latent_dim, backbone, activation, channels)
        self.dropout = nn.Dropout(dropout)  # no weights: the encoder file stays the same
        self.decoder = build_decoder(latent_dim, decoder, channels, activation)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.encoder(x)     # [B, 3, 64, 64] -> [B, latent_dim]
        return self.decoder(self.dropout(h))  # [B, latent_dim] -> [B, 3, 64, 64]

    def add_noise(self, x, generator=None):
        """Corrupt the normalized images x [B, 3, 64, 64]: the noise, then the second noise if any.
        generator=None: new noise every call (training). Seeded: the same noise (val, test)."""
        x = self._corrupt(x, self.hparams.noise_type, self.hparams.noise_level, generator)
        if self.hparams.noise_type2 != "none":
            x = self._corrupt(x, self.hparams.noise_type2, self.hparams.noise_level2, generator)  # stacked
        return x

    def _corrupt(self, x, noise_type, noise_level, generator):
        """One noise of NOISE_TYPES at noise_level."""
        if noise_type == "gaussian":
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            return x + noise_level * noise
        if noise_type == "patch_mask":
            p = self.hparams.patch_size
            r = torch.rand((x.size(0), 1, x.size(2) // p, x.size(3) // p), generator=generator, device=x.device)  # [B, 1, 64/p, 64/p]
            keep = (r >= noise_level).repeat_interleave(p, 2).repeat_interleave(p, 3)  # one number per patch -> [B, 1, 64, 64]
            return x * keep  # the picked patches -> 0, the mean color
        if noise_type == "block":
            side = round(x.size(2) * noise_level ** 0.5)  # the square covers noise_level of the image
            top, left = torch.randint(0, x.size(2) - side + 1, (2, x.size(0), 1, 1, 1), generator=generator, device=x.device)
            rows = torch.arange(x.size(2), device=x.device).view(1, 1, -1, 1)
            cols = torch.arange(x.size(3), device=x.device).view(1, 1, 1, -1)
            hidden = (rows >= top) & (rows < top + side) & (cols >= left) & (cols < left + side)  # [B, 1, 64, 64]
            return x * ~hidden  # the square -> 0, the mean color
        # mask and salt_pepper: one random number per pixel, shared by the 3 channels
        r = torch.rand((x.size(0), 1, x.size(2), x.size(3)), generator=generator, device=x.device, dtype=x.dtype)  # [B, 1, 64, 64]
        if noise_type == "mask":
            return x * (r >= noise_level)  # the picked pixels -> 0, the mean color
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, recon, x, x_noisy = self._common_step(batch, self._fixed_generator(batch_idx))
        if dataloader_idx == 1:  # the test set, for the test curve only
            self.log("testcurve_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
            return loss
        # add_dataloader_idx=False keeps the name "val_loss" with two loaders.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
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
        grid = torchvision.utils.make_grid(images, nrow=n, normalize=True)  # rescaled to [0, 1] for display
        self.logger.experiment.add_image(tag, grid, self.global_step)

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.hparams.optimizer](self.parameters(), lr=self.hparams.learning_rate,
                                                       weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)  # one step per epoch
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


class EncoderClassifier(pl.LightningModule):
    """Encoder plus a new linear head: linear probe, fine-tuning and transfer.

    encoder_path: the state_dict saved after pretraining (None = random-init baseline).
    freeze_encoder=True trains only the head (linear probe); False trains everything.
    finetune_from (ResNet-18, e.g. "layer3"): the encoder layers before it stay frozen.
    dropout: the share of h zeroed in training, before the head (nn.Dropout; off in val and test).
    weight_decay, head_lr_mult, optimizer: the optimizer (OPTIMIZERS), its weight decay, and the
    head's lr as a multiple of the encoder's. activation: of the encoder (ACTIVATIONS), None: its own.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False,
                 backbone="small", label_smoothing=0.0, finetune_from=None, dropout=0.0, weight_decay=0.0,
                 head_lr_mult=1.0, optimizer="adamw", activation=None, channels=(32, 64)):
        super().__init__()
        self.save_hyperparameters()  # the arguments -> self.hparams

        self.encoder = build_encoder(latent_dim, backbone, activation, channels)
        if encoder_path is not None:
            # Weights from pretraining
            state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict)

        self.dropout = nn.Dropout(dropout)  # no weights: old checkpoints still load
        self.head = nn.Linear(latent_dim, num_classes)  # [B, latent_dim] -> [B, num_classes]

        # A frozen encoder stays in eval mode (BatchNorm stats); see train() below.
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Partial fine-tuning: freeze the encoder layers that come before finetune_from.
        self.frozen_layers = []
        if finetune_from is not None:
            names = [name for name, _ in self.encoder.named_children()]
            if finetune_from not in names:
                raise ValueError(f"finetune_from={finetune_from!r} is not a layer of the encoder: {names}")
            self.frozen_layers = [getattr(self.encoder, name) for name in names[:names.index(finetune_from)]]
            for layer in self.frozen_layers:
                layer.eval()
                for p in layer.parameters():
                    p.requires_grad = False

        # label_smoothing: the target puts 1 - label_smoothing on the true class.
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # The metrics: top-1 and top-5 accuracy (the assignment), and macro precision,
        # recall, F1 and AUROC (one value per class, then their mean).
        metrics = torchmetrics.MetricCollection({
            # average="micro": correct images / all images
            "accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes, average="micro"),
            "top5_accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5, average="micro"),
            "precision": torchmetrics.classification.MulticlassPrecision(num_classes, average="macro"),
            "recall": torchmetrics.classification.MulticlassRecall(num_classes, average="macro"),
            "f1": torchmetrics.classification.MulticlassF1Score(num_classes, average="macro"),
            "auroc": torchmetrics.classification.MulticlassAUROC(num_classes, average="macro"),
            # ECE: how far the confidence is from the accuracy (overconfidence), over 15 bins
            "calibration_error": torchmetrics.classification.MulticlassCalibrationError(num_classes, n_bins=15),
        })
        self.val_metrics = metrics.clone(prefix="val_")  # val_accuracy drives EarlyStopping
        self.test_metrics = metrics.clone(prefix="test_")
        # The test set after every epoch, to plot only
        self.testcurve_metrics = metrics.clone(prefix="testcurve_")
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes, average="micro")
        # Rows: the true class, columns: the predicted class
        self.test_confusion = torchmetrics.classification.MulticlassConfusionMatrix(num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.hparams.freeze_encoder:
            self.encoder.eval()
        for layer in self.frozen_layers:  # partial fine-tuning
            layer.eval()
        return self

    def forward(self, x):
        h = self.encoder(x)  # [B, 3, 64, 64] -> [B, latent_dim]
        return self.head(self.dropout(h))  # [B, latent_dim] -> [B, num_classes]

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.train_accuracy(scores, y)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": self.train_accuracy},
            on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, scores, y = self._common_step(batch, batch_idx)
        # dataloader_idx 1 is the test set, for the test curve only.
        # float(): with 16-mixed the scores are float16, too coarse for AUROC.
        # add_dataloader_idx=False keeps the names EarlyStopping and ModelCheckpoint watch.
        split, metrics, prog_bar = ("testcurve", self.testcurve_metrics, False) if dataloader_idx == 1 else ("val", self.val_metrics, True)
        metrics.update(scores.float(), y)
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, prog_bar=prog_bar, add_dataloader_idx=False)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=prog_bar, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.test_metrics.update(scores.float(), y)
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
        # Only the trainable parameters (linear probe: the head); the head has its own lr.
        lr = self.hparams.learning_rate
        encoder = [p for p in self.encoder.parameters() if p.requires_grad]
        groups = [{"params": self.head.parameters(), "lr": lr * self.hparams.head_lr_mult}]
        if encoder:
            groups.append({"params": encoder, "lr": lr})
        return OPTIMIZERS[self.hparams.optimizer](groups, lr=lr, weight_decay=self.hparams.weight_decay)


class SupervisedEncoder(EncoderClassifier):
    """The same encoder and head, trained from scratch with labels (supervised baseline):
    EncoderClassifier with no loaded weights and nothing frozen.
    **ignored takes the hyperparameters load_from_checkpoint passes back."""

    def __init__(self, learning_rate, num_classes, latent_dim, backbone="small", label_smoothing=0.0, dropout=0.0,
                 weight_decay=0.0, optimizer="adamw", activation=None, channels=(32, 64), **ignored):
        super().__init__(learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False,
                         backbone=backbone, label_smoothing=label_smoothing, dropout=dropout, weight_decay=weight_decay,
                         optimizer=optimizer, activation=activation, channels=channels)
