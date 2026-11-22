from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

import config
from knn import knn_evaluate


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar.

    The validation bar is drawn on a second line, and Colab can't redraw a second
    line in place, so every update would be printed as a new line.
    Validation metrics still appear at the end of the main epoch bar.
    The bar is created disabled: a bar disabled only after it was created has
    already drawn itself, and in a notebook that left an empty "Validation:" bar
    after every epoch.
    """

    def init_validation_tqdm(self):
        return Tqdm(disable=True)


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    """Logs one point per epoch under "epoch/...", with the epoch number as the x-axis.

    Lightning's own charts use the training step as the x-axis. Pretraining uses the
    full training set; the target runs use fewer labeled images, so their curves
    don't line up. With the epoch as the x-axis, all modes can be compared on one chart.
    """

    def _log_metrics(self, trainer, names):
        metrics = {}
        for name in names:
            # with on_step=True Lightning stores the epoch value as "<name>_epoch"
            value = trainer.callback_metrics.get(f"{name}_epoch", trainer.callback_metrics.get(name))
            if value is not None:
                metrics[f"epoch/{name}"] = value.item()

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(trainer, ["train_loss", "train_accuracy"])

    def on_validation_end(self, trainer, pl_module):
        # Skip the sanity check and the extra trainer.validate() after training
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._log_metrics(
            trainer,
            ["val_loss", "val_accuracy", "val_top5_accuracy", "val_f1", "val_auroc", "val_knn_top1", "val_knn_top5"],
        )


class ConfusionMatrixCallback(Callback):
    """After the test of a classifier: its confusion matrix on the test set, as a CSV,
    a figure (also in TensorBoard) and the most confused class pairs
    (analysis.save_confusion_matrix). EncoderClassifier fills it in test_step."""

    def on_test_epoch_end(self, trainer, pl_module):
        # Imported here, not at the top: analysis.py sets matplotlib to draw into files
        # only (Agg), and a notebook that imports callbacks would stop showing its plots.
        from analysis import save_confusion_matrix

        confusion = pl_module.test_confusion.compute()  # [num_classes, num_classes]
        pl_module.test_confusion.reset()
        save_confusion_matrix(confusion, trainer.datamodule.class_names(), trainer.logger)


class KNNCallback(Callback):
    """k-NN accuracy of the encoder's h, logged as val_knn_top1 / val_knn_top5
    after every validation, and test_knn_top1 / test_knn_top5 after the test.

    The DAE is trained on reconstruction, but it is judged by how useful h is.
    So ModelCheckpoint watches val_knn_top1 and keeps the best epoch (assignment:
    model selection by val k-NN, not by reconstruction). The memory bank is the
    training split; bank_size=None uses all of it (see config.KNN_BANK_SIZE).
    """

    def __init__(self, k=None, bank_size=None):
        super().__init__()
        # Read config here, when the callback is created, not when this file is imported.
        self.k = k if k is not None else config.KNN_K
        if bank_size is None:
            # A search trial uses the smaller bank (config.SEARCH_KNN_BANK_SIZE); final runs use all of train.
            bank_size = config.SEARCH_KNN_BANK_SIZE if config.SEARCH else config.KNN_BANK_SIZE
        self.bank_size = bank_size

    def _log_knn(self, trainer, pl_module, split):
        top1, top5 = knn_evaluate(pl_module.encoder, trainer.datamodule, self.k, split, self.bank_size)
        pl_module.log(f"{split}_knn_top1", top1)
        pl_module.log(f"{split}_knn_top5", top5)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # the 2 check batches before training: h is still random
            return
        self._log_knn(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_knn(trainer, pl_module, "test")
