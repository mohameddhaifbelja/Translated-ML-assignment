import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from transformers import BertModel, AutoModel, AdamW


class BertClassifier(pl.LightningModule):
    def __init__(self, dropout=0.1, lr=1e-4, cls_threshold=0.5):
        super(BertClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

        self.lr = lr
        self.cls_threshold = cls_threshold
        self.loss = nn.BCEWithLogitsLoss()

        # Train Metrics
        self.trf1 = torchmetrics.classification.BinaryF1Score(threshold=cls_threshold, task="binary")
        self.trpr = torchmetrics.classification.BinaryPrecision(threshold=cls_threshold, task="binary")
        self.trrl = torchmetrics.classification.BinaryRecall(threshold=cls_threshold, task="binary")

        # Val Metrics
        self.vlf1 = torchmetrics.classification.BinaryF1Score(threshold=cls_threshold, task="binary")
        self.vlpr = torchmetrics.classification.BinaryPrecision(threshold=cls_threshold, task="binary")
        self.vlrl = torchmetrics.classification.BinaryRecall(threshold=cls_threshold, task="binary")

        # Test Metrics
        self.tsf1 = torchmetrics.classification.BinaryF1Score(threshold=cls_threshold, task="binary")
        self.tspr = torchmetrics.classification.BinaryPrecision(threshold=cls_threshold, task="binary")
        self.tsrl = torchmetrics.classification.BinaryRecall(threshold=cls_threshold, task="binary")

    def forward(self, input_ids, attention_mask=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

    def get_preds(self, logits):
        return (nn.Sigmoid()(logits).view(-1) > self.cls_threshold).float()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].float()
        attention_mask = batch['attention_mask']
        labels = batch['label'].view(-1).float()

        logits = self(input_ids, attention_mask).view(-1).float()
        loss = self.loss(logits, labels)

        preds = self.get_preds(logits)
        self.trf1(preds, labels)
        self.trpr(preds, labels)
        self.trrl(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', self.trf1, on_epoch=True, prog_bar=True)
        self.log('train_precision', self.trpr, on_epoch=True)
        self.log('train_recall', self.trrl, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].view(-1).float()

        logits = self(input_ids, attention_mask).view(-1).float()
        loss = self.loss(logits, labels)

        preds = self.get_preds(logits)

        self.vlf1(preds, labels)
        self.vlpr(preds, labels)
        self.vlrl(preds, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.vlf1, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.vlpr, on_epoch=True)
        self.log('val_recall', self.vlrl, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'].view(-1).float()

        logits = self(input_ids, attention_mask).view(-1).float()

        preds = self.get_preds(logits)
        self.tsf1.update(preds, labels)
        self.tspr.update(preds, labels)
        self.tsrl.update(preds, labels)

        self.log('test_f1', self.tsf1, on_epoch=True, prog_bar=True)
        self.log('test_precision', self.tspr, on_epoch=True)
        self.log('test_recall', self.tsrl, on_epoch=True)

        return

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
