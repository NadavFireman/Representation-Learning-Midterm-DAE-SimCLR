import torch
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision

# Per-channel mean and std of the Tiny-ImageNet train split (90,000 images, val and
# test not included), on pixels in [0, 1]. Computed once with datamodule.py.
TINY_IMAGENET_MEAN = (0.4804, 0.4482, 0.3976)
TINY_IMAGENET_STD = (0.2764, 0.2688, 0.2816)


class Normalize(nn.Module):
    """(x - mean) / std per channel, as the first layer of the encoder.

    The course code normalized in the transforms. Here the data stays in [0, 1],
    because the decoder ends in a Sigmoid and its clean target must be in [0, 1].
    The encoder still gets centered input, because it normalizes by itself.
    mean and std are buffers, not parameters: they are saved in the encoder's
    state_dict, but the optimizer never changes them. So a saved encoder
    normalizes the same way in k-NN, the linear probe and fine-tuning.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))  # [1, 3, 1, 1], broadcasts over [B, 3, 64, 64]
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std  # [B, 3, 64, 64] -> [B, 3, 64, 64]


def build_encoder(latent_dim):
    """The encoder f_theta: [B, 3, 64, 64] -> h [B, latent_dim].

    The lec3 encoder, for 3x64x64 images instead of 1x28x28. Two stride-2
    convolutions halve the image twice (64 -> 32 -> 16), so the Linear layer reads
    64*16*16 numbers instead of 64*7*7. The course wrote these layers out again in
    every class. Here one function builds them, so every model is sure to get the
    same encoder, and a saved state_dict loads into any of them.
    """
    return nn.Sequential(
        Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),  # [B, 3, 64, 64] -> [B, 3, 64, 64], centered
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, 32, 32, 32]
        nn.ReLU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
        nn.ReLU(),  # [B, 64, 16, 16] -> [B, 64, 16, 16]
        nn.Flatten(),  # [B, 64, 16, 16] -> [B, 16384]
        nn.Linear(64 * 16 * 16, latent_dim),  # [B, 16384] -> [B, latent_dim]
    )


class DenoisingAutoEncoder(pl.LightningModule):
    """Denoising autoencoder: corrupted image in, clean image out.

    The lec3 AutoEncoder with one change in the loss. The input is corrupted,
    and the loss is MSE between the reconstruction and the clean image. To undo
    the corruption, the encoder has to keep what the image shows and drop the
    noise, and that is the representation h we want. No skip connections (not
    a U-Net): everything passes through h.

    noise_type, the three corruptions the assignment lists for a DAE:
        "gaussian":    x + noise_level * N(0, 1), clipped to [0, 1]
        "mask":        a noise_level fraction of the pixels is set to 0 (all 3 channels)
        "salt_pepper": a noise_level fraction of the pixels is set to 0 or 1, half each
    noise_level=0 changes nothing, which is the plain autoencoder of lec3.
    That is the AE baseline, with no separate class.

    The noise is added here and not in the transforms, because the loss needs
    the clean image as well. It is added in val and test too, so val_loss
    measures the same task the model trains on. In val and test the noise of
    batch i always comes from seed i, so every epoch and every trial is scored
    on the same corrupted images and their reconstruction errors can be compared.

    The decoder mirrors the encoder (16 -> 32 -> 64) and ends in a Sigmoid, so
    every pixel is in (0, 1), the range of the clean image. After pretraining
    only the encoder is kept.
    """

    def __init__(self, learning_rate, latent_dim, noise_type, noise_level):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.noise_type = noise_type
        self.noise_level = noise_level

        self.encoder = build_encoder(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),  # [B, latent_dim] -> [B, 16384]
            nn.ReLU(),  # [B, 16384] -> [B, 16384]
            nn.Unflatten(1, (64, 16, 16)),  # [B, 16384] -> [B, 64, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 16, 16] -> [B, 32, 32, 32]
            nn.ReLU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 32, 32] -> [B, 3, 64, 64]
            nn.Sigmoid(),  # [B, 3, 64, 64] -> [B, 3, 64, 64], values in (0, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        h = self.encoder(x)     # [B, 3, 64, 64] -> [B, latent_dim]
        return self.decoder(h)  # [B, latent_dim] -> [B, 3, 64, 64]

    def add_noise(self, x, generator=None):
        """Corrupt x [B, 3, 64, 64] in [0, 1]; the result is still an image in [0, 1].

        generator=None draws new noise every call (training). A seeded generator
        gives the same noise every time (val and test).
        """
        if self.noise_type == "gaussian":
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            return (x + self.noise_level * noise).clamp(0, 1)
        # mask and salt_pepper pick whole pixels: one random number per pixel, shared by the 3 channels
        r = torch.rand((x.size(0), 1, x.size(2), x.size(3)), generator=generator, device=x.device, dtype=x.dtype)  # [B, 1, 64, 64]
        if self.noise_type == "mask":
            return x * (r >= self.noise_level)  # a noise_level fraction of the pixels -> 0
        if self.noise_type == "salt_pepper":
            x = torch.where(r < self.noise_level / 2, torch.zeros_like(x), x)  # pepper: -> 0
            return torch.where(r > 1 - self.noise_level / 2, torch.ones_like(x), x)  # salt: -> 1
        raise ValueError(f"unknown noise_type: {self.noise_type}")

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
        grid = torchvision.utils.make_grid(images, nrow=n)
        self.logger.experiment.add_image(tag, grid, self.global_step)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class EncoderClassifier(pl.LightningModule):
    """Encoder plus a new linear head. Used for linear probe, fine-tuning and transfer.

    encoder_path is the state_dict saved after pretraining. encoder_path=None
    keeps the random initial weights, which is the random-init baseline.
    freeze_encoder=True trains only the head (linear probe).
    False trains the encoder and the head (fine-tuning).

    Logs top-1 and top-5 accuracy, as the assignment asks.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.freeze_encoder = freeze_encoder

        self.encoder = build_encoder(latent_dim)
        if encoder_path is not None:
            # Weights from pretraining. The head below is not in this file.
            state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict)

        self.head = nn.Linear(latent_dim, num_classes)  # [B, latent_dim] -> [B, num_classes]

        # eval() before freezing: a frozen encoder must not update BatchNorm stats
        # or dropout. train() below keeps that, because Lightning calls train()
        # at the start of every epoch and would otherwise undo it.
        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.top5_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        h = self.encoder(x)  # [B, 3, 64, 64] -> [B, latent_dim]
        return self.head(h)  # [B, latent_dim] -> [B, num_classes]

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        x, y = batch
        if batch_idx % 100 == 0 and self.logger is not None:
            grid = torchvision.utils.make_grid(x[:8])  # [8, 3, 64, 64], already in [0, 1]
            self.logger.experiment.add_image("train_images", grid, self.global_step)
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        top5_accuracy = self.top5_accuracy(scores, y)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy, "val_top5_accuracy": top5_accuracy},
            on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        top5_accuracy = self.top5_accuracy(scores, y)
        self.log_dict(
            {"test_loss": loss, "test_accuracy": accuracy, "test_top5_accuracy": top5_accuracy},
            on_step=False, on_epoch=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, 64, 64]
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        # Linear probing: only the head is trainable, so only the head is optimized.
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.lr)


class SupervisedEncoder(EncoderClassifier):
    """The same encoder and head, trained from scratch with labels (supervised baseline).

    The course wrote this class out in full. It is exactly EncoderClassifier with
    no loaded weights and nothing frozen, so here it is that, under its own name.

    EncoderClassifier saves encoder_path=None and freeze_encoder=False in the
    hyperparameters, and load_from_checkpoint passes them back to this __init__.
    **ignored takes them, so a saved SupervisedEncoder can be loaded again;
    the values are always None and False here.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, **ignored):
        super().__init__(learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False)
