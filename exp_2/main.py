import os

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import hydra
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
from torchmetrics.functional import auroc


class CustumAttentionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int,
        text_column_name: str,
        label_column_name: str,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_columm_name = text_column_name
        self.label_column_name = label_column_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        df_row = self.df.iloc[index]
        text = df_row[self.text_columm_name]
        labels = df_row[self.label_column_name].astype(int)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt",
        )

        return encoding["input_ids"].flatten(), torch.tensor(labels)


class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        batch_size,
        max_length,
        text_column_name: str = "text",
        label_column_name: str = "label",
        pretrained_model="cl-tohoku/bert-base-japanese-whole-word-masking",
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.text_columm_name = text_column_name
        self.label_column_name = label_column_name

    def setup(self):
        self.train_dataset = CustumAttentionDataset(
            self.train_df,
            self.tokenizer,
            self.max_length,
            self.text_columm_name,
            self.label_column_name,
        )
        self.vaild_dataset = CustumAttentionDataset(
            self.valid_df,
            self.tokenizer,
            self.max_length,
            self.text_columm_name,
            self.label_column_name,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


class CustumBert(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        d_model: int,
        learning_rate: float,
        max_length: int,
        pretrained_model="cl-tohoku/bert-base-japanese-whole-word-masking",
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes

        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = self.bert(inputs)[0]
        preds = self.classifier(outputs[:, 0, :])
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(inputs=x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(inputs=x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "batch_preds": y_hat, "batch_labels": y}

    def training_epoch_end(self, outputs, mode="train"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

        epoch_auroc = auroc(epoch_y_hats, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

        epoch_auroc = auroc(epoch_y_hats, epoch_labels, num_classes=self.n_classes)
        self.log(f"{mode}_auroc", epoch_auroc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def make_callbacks(min_delta, patience, checkpoint_path):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    print(f"cwd:{cwd}")
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
        log_model=True,
    )
    print(f"wandb logger :{wandb_logger.experiment.dir}")
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )
    wandb_logger.log_hyperparams(cfg)
    df = pd.read_csv(cfg.path.data_file_name, sep="\t").dropna().reset_index(drop=True)
    df[cfg.training.label_column_name] = np.argmax(df.iloc[:, 2:].values, axis=1)
    df[[cfg.training.text_column_name, cfg.training.label_column_name]]
    train, test = train_test_split(df, test_size=cfg.training.test_size, shuffle=True)
    data_module = CreateDataModule(
        train,
        test,
        cfg.training.batch_size,
        cfg.model.max_length,
        cfg.training.text_column_name,
    )
    data_module.setup()

    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )
    model = CustumBert(
        n_classes=cfg.model.n_classes,
        d_model=cfg.model.d_model,
        learning_rate=cfg.training.learning_rate,
        max_length=cfg.model.max_length,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        gpus=1,
        progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)
    wandb_logger.finalize()


if __name__ == "__main__":
    main()
