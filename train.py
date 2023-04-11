from model import BertClassifier
from data import PLData
from transformers import BertTokenizer
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle

def train(seq_len):
    trainer = Trainer(
        gpus=-1,
        max_epochs=20,
        logger=[CSVLogger(save_dir="logs/"), TensorBoardLogger(save_dir="logs/")],
        callbacks=[ModelCheckpoint(save_top_k=1, save_last=1), TQDMProgressBar(refresh_rate=20)],
    )

    model = BertClassifier()
    data = PLData(seq_len=seq_len, trainfile="data/train.csv", testfile="data/test.csv", language="Italian", batchsize=32)
    data.setup(stage="")

    trainer.fit(model, data)
    #trainer.test(model, data)
    file = open(f"{seq_len}_ml_weights.pkl", "wb")
    pickle.dump(model, file)

def test_model(model_path, seq_len):
    trainer = Trainer()

    # model = BertClassifier()
    # model.load_from_checkpoint(model_path)
    file = open(model_path,'rb')
    model = pickle.load(file)
    data = PLData(seq_len=seq_len, trainfile="data/train.csv", testfile="data/test.csv", language="Italian", batchsize=32)
    data.setup(stage="")

    trainer.test(model, data)

if __name__ == "__main__":
    import time
    now = time.time()
    seed_everything(42)
    #train(seq_len=128)
    test_model(model_path="128_weights.pkl", seq_len=128)
    print(time.time() - now)