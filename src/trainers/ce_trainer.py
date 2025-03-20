import pytorch_lightning as pl
from src.models.resnet import CustomResNet
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
import itertools
import io
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pytorch_lightning.loggers import CometLogger
from src.metrics import mmd, class_wise_mmd, different_class_mmd

class CrossEntropyTrainer(pl.LightningModule):
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        learning_rate: float = 0.001, 
        num_domains: int = 1, 
        plot_pca: bool = False, 
        plot_tsne: bool = False
        ):
        
        super().__init__()

        if model_name == "resnet18":
            self.model = CustomResNet(
                block=models.resnet.BasicBlock,
                layers=[2, 2, 2, 2],
                inplanes=64,
                num_classes=num_classes,
                in_kernel_size=3
                )
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

        self.plot_pca = plot_pca
        self.plot_tsne = plot_tsne
                
        # Replace the final fully connected layer to match the number of classes
        self.num_classes = num_classes
        self.num_domains = num_domains

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.init_train_metrics()
        self.init_validation_metrics()
        self.init_test_metrics()

    def init_train_metrics(self):
        """Initializes training metrics such as accuracy and confusion matrix components."""

        # accuracy
        self.acc_train = []

        # accumulate for metrics
        self.train_feats = []
        self.train_domains = []
        self.train_y = []
        self.train_y_pred = []

    def init_validation_metrics(self):
        """Initializes validation metrics including loss, accuracy, and confusion matrix components."""

        # loss
        self.loss_val = []

        # accumulate for metrics
        self.val_feats = []
        self.val_domains = []
        self.val_y = []
        self.val_y_pred = []

    def init_test_metrics(self):
        """Initializes test metrics including loss, accuracy, and confusion matrix components."""

        # loss
        self.loss_test = []

        # accumulate for metrics
        self.test_feats = []
        self.test_domains = []
        self.test_y = []
        self.test_y_pred = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images, labels, domains = batch
        outputs, features = self.model.forward_with_features(images)
        features = features.cpu().detach()
        loss = self.criterion(outputs, labels)

        # prediction
        outputs = outputs.argmax(dim=1)

        # accumulate features by domain
        self.train_feats.append(features)
        self.train_domains.append(domains)

        # accumulate for metrics
        self.train_y.append(labels.cpu().detach())
        self.train_y_pred.append(outputs.cpu().detach())

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single validation step.
        
        Args:
            batch: A batch of validation data.
            batch_idx: The index of the batch.
        """
        images, labels, domains = batch
        outputs, features = self.model.forward_with_features(images)
        features = features.cpu().detach()

        # loss
        loss = self.criterion(outputs, labels)
        self.loss_val.append(loss.detach().cpu())

        # prediction
        outputs = outputs.argmax(dim=1)
    
        # accumulate features by domain
        self.val_feats.append(features)
        self.val_domains.append(domains)

        self.val_y.append(labels.cpu().detach())
        self.val_y_pred.append(outputs.cpu().detach())

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single test step.
        
        Args:
            batch: A batch of test data.
            batch_idx: The index of the batch.
        """
        images, labels, domains = batch
        outputs, features = self.model.forward_with_features(images)
        features = features.cpu().detach()

        # loss
        loss = self.criterion(outputs, labels)
        self.loss_test.append(loss.detach().cpu())
    
        # prediction
        outputs = outputs.argmax(dim=1)

        self.test_feats.append(features)
        self.test_domains.append(domains)

        self.test_y.append(labels.cpu().detach())
        self.test_y_pred.append(outputs.cpu().detach())
    
    def get_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, Any]:
        """
        Computes accuracy, F1 scores, and the confusion matrix.
        
        Args:
            y_pred: Predicted labels.
            y_true: Ground truth labels.
        
        Returns:
            A dictionary containing accuracy, F1-macro, F1-weighted, and confusion matrix.
        """

        # accuracy
        acc = (y_pred == y_true).float().mean().item()

        # f1 scores
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        out = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cm": cm
        }

        return out
    
    def get_mean_mmd_metrics(self, features: torch.Tensor, domains: torch.Tensor, y_true: torch.Tensor):

        features, domains = features.numpy(), domains.numpy()
        domain_indices = list(range(self.num_domains))
        domain_combinations = list(itertools.combinations(domain_indices, 2))
        
        mmd_accum = []
        cmmd_accum = []
        dcmmd_accum = []
        for dom1, dom2 in domain_combinations:

            feats1 = features[domains == dom1]
            feats2 = features[domains == dom2]
            y_true1 = y_true[domains == dom1]
            y_true2 = y_true[domains == dom2]

            mmd_accum.append(mmd(feats1, feats2))
            cmmd_accum.append(class_wise_mmd(feats1, feats2, y_true1, y_true2))
            dcmmd_accum.append(different_class_mmd(feats1, feats2, y_true1, y_true2))

        return {"mmd": np.mean(mmd_accum), "cmmd": np.mean(cmmd_accum), "dcmmd": np.mean(dcmmd_accum)}

    def on_train_epoch_end(self):
        """Computes and logs training metrics at the end of an epoch."""
        
        y_pred = torch.cat(self.train_y_pred)
        y_true = torch.cat(self.train_y)
        domains = torch.cat(self.train_domains)
        features = torch.cat(self.train_feats)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))

        for name, value in metrics.items():
            if name == "cm":
                pass
                # TODO: differentiate train and valid cm in comet interface (title is not enough)
                self.train_cm = value
            else:
                self.log(f"train_{name}", value, on_step=False, on_epoch=True)
        
        if self.plot_pca:
            pca_fig = self.generate_pca_plot(features.squeeze(), y_true)
            self.log_figure(pca_fig, fig_name = "Train PCA (by label)")
    
            pca_fig = self.generate_pca_plot(features.squeeze(), domains)
            self.log_figure(pca_fig, fig_name = "Train PCA (by domain)")

        if self.plot_tsne:
            tsne_fig = self.generate_tsne_plot(features.squeeze(), y_true)
            self.log_figure(tsne_fig, fig_name = "Train t-SNE (by label)")

            tsne_fig = self.generate_tsne_plot(features.squeeze(), domains)
            self.log_figure(tsne_fig, fig_name = "Train t-SNE (by domain)")

        # reset metrics
        self.init_train_metrics()

    def on_validation_epoch_end(self):
        """Computes and logs validation metrics at the end of an epoch."""

        y_pred = torch.cat(self.val_y_pred)
        y_true = torch.cat(self.val_y)
        domains = torch.cat(self.val_domains)
        features = torch.cat(self.val_feats)

        loss = np.mean(self.loss_val)
        self.log(f"val_loss", loss, on_step=False, on_epoch=True)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))

        for name, value in metrics.items():
            if name == "cm":
                if isinstance(self.logger, CometLogger):
                    self.logger.experiment.log_confusion_matrix(matrix=value, step=self.current_epoch, title="val_cm")
                self.val_cm = value
            else:
                self.log(f"val_{name}", value, on_step=False, on_epoch=True)

        if self.plot_pca:
            pca_fig = self.generate_pca_plot(features.squeeze(), y_true)
            self.log_figure(pca_fig, fig_name = "Valid PCA (by label)")
    
            pca_fig = self.generate_pca_plot(features.squeeze(), domains)
            self.log_figure(pca_fig, fig_name = "Valid PCA (by domain)")

        if self.plot_tsne:
            tsne_fig = self.generate_tsne_plot(features.squeeze(), y_true)
            self.log_figure(tsne_fig, fig_name = "Valid t-SNE (by label)")

            tsne_fig = self.generate_tsne_plot(features.squeeze(), domains)
            self.log_figure(tsne_fig, fig_name = "Valid t-SNE (by domain)")

        # reset metrics
        self.init_validation_metrics()

    def on_test_epoch_end(self):
        """Computes and logs test metrics at the end of an epoch."""

        y_pred = torch.cat(self.test_y_pred)
        y_true = torch.cat(self.test_y)
        domains = torch.cat(self.test_domains)
        features = torch.cat(self.test_feats)

        loss = np.mean(self.loss_test)
        self.log(f"test_loss", loss, on_step=False, on_epoch=True)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))
        
        for name, value in metrics.items():
            if name == "cm":
                self.test_cm = value
            else:
                self.log(f"test_{name}", value, on_step=False, on_epoch=True)

        if self.plot_pca:
            pca_fig = self.generate_pca_plot(features.squeeze(), y_true)
            self.log_figure(pca_fig, fig_name = "Test PCA")

        if self.plot_tsne:
            tsne_fig = self.generate_tsne_plot(features.squeeze(), y_true)
            self.log_figure(tsne_fig, fig_name = "Test t-SNE")

        # reset metrics
        self.init_test_metrics()

        return metrics

    def configure_optimizers(self):

        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def generate_pca_plot(self, embeddings, labels):
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="Paired", alpha=0.7)

        # Add colorbar
        plt.colorbar(scatter, label="Class")

        # Create a legend
        unique_labels = np.unique(labels)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=8) for label in unique_labels]
        ax.legend(handles, unique_labels, title="Classes", loc="best")

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Embeddings")

        return fig

    def generate_tsne_plot(self, embeddings, labels):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        reduced_embeddings = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="Paired", alpha=0.7)
        plt.colorbar(scatter, label="Class")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE of Embeddings")

        return fig

    def log_figure(self, fig, fig_name = "PCA plot"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        self.logger.experiment.log_figure(figure_name=fig_name, figure=fig)