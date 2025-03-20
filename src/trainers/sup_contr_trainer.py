from typing import List
import pytorch_lightning as pl
from src.models.resnet import CustomResNet
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Dict, Any
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, f1_score
from pytorch_lightning.loggers import CometLogger
from src.trainers.ce_trainer import CrossEntropyTrainer
from src.schedulers import StepVariableScheduler, LinearVariableScheduler, MultiStepVariableScheduler
from third_party.sup_contr_loss import SupConLoss
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = 2 * embedding_dim
        
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class SupContrTrainer(CrossEntropyTrainer):

    def __init__(
        self, 
        model_name: int, 
        num_classes: int, 
        learning_rate: float = 0.001, 
        num_domains: int = 1, 
        contr_temp: float = 0.5,
        use_contr_temp_scheduler: bool = False,
        final_contr_temp: float = 0.5,
        contr_temp_steps: int = 0,
        contr_weight: float = 1.0,
        use_contr_weight_scheduler: bool = False,
        contr_weight_scheduler_step: int = 0,
        contr_weight_final: float = 0.0,
        use_freeze_weights_scheduler: bool = False,
        freeze_weights_scheduler_values: List[bool] = [],
        freeze_weights_scheduler_steps: List[int] = [],
        plot_pca: bool = True,
        plot_tsne: bool = True,
        use_projector: bool = True
        ):

        super().__init__(model_name, num_classes, learning_rate=learning_rate, num_domains=num_domains, plot_pca=plot_pca, plot_tsne=plot_tsne)

        self.ce_criterion = nn.CrossEntropyLoss()
        self.contr_criterion = SupConLoss(temperature=contr_temp)
        if use_projector:
            embedding_dim = self.model.get_embedding_dim()
            self.projector = Projector(embedding_dim)
        else:
            self.projector = None

        # contrastive weight
        self.contr_weight = contr_weight
        self.use_contr_weight_scheduler = use_contr_weight_scheduler
        self.contr_weight_scheduler = StepVariableScheduler(contr_weight, contr_weight_final, contr_weight_scheduler_step)

        # temperature
        self.use_contr_temp_scheduler = use_contr_temp_scheduler
        self.contr_temp_scheduler = LinearVariableScheduler(contr_temp, final_contr_temp, contr_temp_steps)

        # freeze weights scheduler
        self.use_freeze_weights_scheduler = use_freeze_weights_scheduler
        self.weights_frozen = False
        self.freeze_weights_scheduler = MultiStepVariableScheduler(False, freeze_weights_scheduler_values, freeze_weights_scheduler_steps)

    def init_train_metrics(self):

        super().init_train_metrics()
        self.train_feats_sup_contr = []

    def init_validation_metrics(self):

        super().init_validation_metrics()

        self.contr_loss_val = []
        self.ce_loss_val = []
        self.val_feats_sup_contr = []

    def init_test_metrics(self):
        
        super().init_test_metrics()

        self.contr_loss_test = []
        self.ce_loss_test = []
        self.test_feats_sup_contr = []

    def training_step(self, batch, batch_idx):

        images, labels, domains = batch
        predictions, features = self.model.forward_with_features(images)

        if self.projector is None:
            sup_contr_features = features
        else:
            sup_contr_features = self.projector(features)

        sup_contr_features = sup_contr_features.unsqueeze(1)
        sup_contr_features = F.normalize(sup_contr_features, p=2, dim=2)

        # loss calculation
        #labels = labels.long()
        contr_loss = self.contr_criterion(sup_contr_features, labels)
        ce_loss = self.ce_criterion(predictions, labels)

        self.log("train_ce_loss", ce_loss)
        self.log("train_contr_loss", contr_loss)
        self.log("contr_weight", self.contr_weight)
        self.log("temperature", self.contr_criterion.temperature)
        self.log("weights_frozen", int(self.weights_frozen))

        loss = (1 - self.contr_weight) * ce_loss + self.contr_weight * contr_loss

        # prediction
        predictions = predictions.argmax(dim=1)

        # accumulate features by domain
        self.train_feats.append(features.cpu().detach())
        self.train_feats_sup_contr.append(sup_contr_features.cpu().detach())
        self.train_domains.append(domains)

        # accumulate for metrics
        self.train_y.append(labels.cpu().detach())
        self.train_y_pred.append(predictions.cpu().detach())

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single validation step.
        
        Args:
            batch: A batch of validation data.
            batch_idx: The index of the batch.
        """
        images, labels, domains = batch
        predictions, features = self.model.forward_with_features(images)
    
        if self.projector is None:
            sup_contr_features = features
        else:
            sup_contr_features = self.projector(features)

        sup_contr_features = sup_contr_features.unsqueeze(1)
        sup_contr_features = F.normalize(sup_contr_features, p=2, dim=2)

        # loss
        contr_loss = self.contr_criterion(sup_contr_features, labels)
        ce_loss = self.ce_criterion(predictions, labels)
        loss = (1 - self.contr_weight) * ce_loss + self.contr_weight * contr_loss
    
        self.loss_val.append(loss.detach().cpu())
        self.contr_loss_val.append(contr_loss.detach().cpu())
        self.ce_loss_val.append(ce_loss.detach().cpu())

        # prediction
        predictions = predictions.argmax(dim=1)
    
        # accumulate features by domain
        self.val_feats.append(features.cpu().detach())
        self.val_feats_sup_contr.append(sup_contr_features.cpu().detach())
        self.val_domains.append(domains)

        self.val_y.append(labels.cpu().detach())
        self.val_y_pred.append(predictions.cpu().detach())

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single test step.
        
        Args:
            batch: A batch of test data.
            batch_idx: The index of the batch.
        """
        images, labels, domains = batch
        predictions, features = self.model.forward_with_features(images)

        if self.projector is None:
            sup_contr_features = features
        else:
            sup_contr_features = self.projector(features)

        sup_contr_features = sup_contr_features.unsqueeze(1)
        sup_contr_features = F.normalize(sup_contr_features, p=2, dim=2)

        # loss
        contr_loss = self.contr_criterion(sup_contr_features, labels)
        ce_loss = self.ce_criterion(predictions, labels)
        loss = (1 - self.contr_weight) * ce_loss + self.contr_weight * contr_loss
    
        self.loss_test.append(loss.detach().cpu())
        self.contr_loss_test.append(contr_loss.detach().cpu())
        self.ce_loss_test.append(ce_loss.detach().cpu())
    
        # prediction
        predictions = predictions.argmax(dim=1)

        self.test_feats.append(features.cpu().detach())
        self.test_feats_sup_contr.append(sup_contr_features.cpu().detach())
        self.test_domains.append(domains)

        self.test_y.append(labels.cpu().detach())
        self.test_y_pred.append(predictions.cpu().detach())

    def configure_optimizers(self, train_only_linear_layer: bool = False):

        if train_only_linear_layer:
            params = self.model.parameters_fc()
        else:
            params = self.model.parameters()

        return optim.Adam(params, lr=self.learning_rate)

    #def on_train_epoch_end(self):
    def on_train_epoch_start(self):

        super().on_train_epoch_start()

        self.contr_weight_scheduler.step()
        self.contr_temp_scheduler.step()
        self.freeze_weights_scheduler.step()

        if self.use_contr_weight_scheduler:
            self.contr_weight = self.contr_weight_scheduler.get_value()
        
        if self.use_contr_temp_scheduler:
            temp = self.contr_temp_scheduler.get_value()
            self.contr_criterion = SupConLoss(temperature=temp)

        if self.use_freeze_weights_scheduler:
            freeze_weights = self.freeze_weights_scheduler.get_value()
            print(freeze_weights, self.weights_frozen)
            if freeze_weights != self.weights_frozen:
                print("got in!")
                self.model.freeze_or_unfreeze_except_fc(train_params=not freeze_weights)
                # if some weights are (un)frozen, the optimizer needs to be re-defined
                self.trainer.optimizers = [self.configure_optimizers(freeze_weights)]
                self.weights_frozen = freeze_weights

    def on_train_epoch_end(self):
        """Computes and logs training metrics at the end of an epoch."""
        
        y_pred = torch.cat(self.train_y_pred)
        y_true = torch.cat(self.train_y)
        domains = torch.cat(self.train_domains)
        features = torch.cat(self.train_feats)
        features_sup_contr = torch.cat(self.train_feats_sup_contr)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))

        _sup_contr_mmed_metrics = self.get_mean_mmd_metrics(features_sup_contr, domains, y_true)
        _sup_contr_mmed_metrics = {"sup_contr_" + k: v for k,v in _sup_contr_mmed_metrics.items()}
        metrics.update(_sup_contr_mmed_metrics)

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
        features_sup_contr = torch.cat(self.val_feats_sup_contr)

        loss = np.mean(self.loss_val)
        self.log(f"val_loss", loss, on_step=False, on_epoch=True)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))
        

        _sup_contr_mmed_metrics = self.get_mean_mmd_metrics(features_sup_contr, domains, y_true)
        _sup_contr_mmed_metrics = {"sup_contr_" + k: v for k,v in _sup_contr_mmed_metrics.items()}
        metrics.update(_sup_contr_mmed_metrics)

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
        features_sup_contr = torch.cat(self.test_feats_sup_contr)

        loss = np.mean(self.loss_test)
        self.log(f"test_loss", loss, on_step=False, on_epoch=True)

        metrics = self.get_metrics(y_pred, y_true)
        metrics.update(self.get_mean_mmd_metrics(features, domains, y_true))

        _sup_contr_mmed_metrics = self.get_mean_mmd_metrics(features_sup_contr, domains, y_true)
        _sup_contr_mmed_metrics = {"sup_contr_" + k: v for k,v in _sup_contr_mmed_metrics.items()}
        metrics.update(_sup_contr_mmed_metrics)
        
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