import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer


class LangDetectionDataset(Dataset):
    def __init__(self,seq_len, file, language, tokenizer):
        df = pd.read_csv(file)
        self.texts = df['Text'].values
        df["label"] = df["Language"].apply(lambda x: 1 if x == language else 0)

        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = seq_len  # we fix the length of the sequence

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class PLData(pl.LightningDataModule):
    def __init__(self, seq_len, trainfile, testfile, batchsize, language):
        super().__init__()
        self.trainfile = trainfile
        self.testfile = testfile
        self.batchsize = batchsize
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language = language
        self.seq_len = seq_len

    def setup(self, stage: str) -> None:
        trainset = LangDetectionDataset(seq_len=self.seq_len, file=self.trainfile, tokenizer=self.tokenizer, language=self.language)
        train_size = int(0.9 * len(trainset))
        val_size = len(trainset) - train_size
        self.train_data, self.val_data = random_split(trainset, [train_size, val_size])
        self.test_data = LangDetectionDataset(seq_len=self.seq_len, file=self.testfile, tokenizer=self.tokenizer, language=self.language)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batchsize, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batchsize, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batchsize, num_workers=4)

if __name__ == "__main__":
    pldata = PLData(seq_len=512, trainfile="data/train.csv", testfile="data/test.csv", batchsize=32, language="Italian")
    pldata.setup(stage="")
    train_positive = 0
    train_negative = 0
    test_positive = 0
    test_negative = 0
    val_positive = 0
    val_negative = 0
    for batch in pldata.train_dataloader():
        for l in batch['label']:
            if l == 0:
                train_negative+=1
            else:
                train_positive+=1
    for batch in pldata.val_dataloader():
        for l in batch['label']:
            if l == 0:
                val_negative+=1
            else:
                val_positive+=1

    for batch in pldata.test_dataloader():
        for l in batch['label']:
            if l == 0:
                test_negative+=1
            else:
                test_positive+=1

    print(f'TRAIN: {train_positive} ,{train_negative}')
    print(f'VAL: {val_positive} ,{val_negative}')
    print(f'TEST: {test_positive} ,{test_negative}')
