import pytorch_lightning as pl
import torchmetrics

BATCH_SIZE = 256
NUM_EPOCHS = 180
LEARNING_RATE = 0.001
NUM_WORKERS = 8

import torch
from hiq.vis import print_model

pytorch_model = torch.hub.load(
    "pytorch/vision:v0.11.0", "mobilenet_v3_small", weights=None
)

pytorch_model.classifier[-1] = torch.nn.Linear(
    in_features=1024, out_features=10
)

#print_model(pytorch_model)

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # Do another forward pass in .eval() mode to compute accuracy
        # while accountingfor Dropout, BatchNorm etc. behavior
        # during evaluation (inference)
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()

        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        #import pudb; pu.db
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log(
            "valid_acc",
            self.valid_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# 
# ## Setting up the dataset

# 
# - In this section, we are going to set up our dataset.

# 
# ### Inspecting the dataset

#
# %load ../code_dataset/dataset_cifar10_check.py
from collections import Counter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


train_dataset = datasets.CIFAR10(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=True,
    shuffle=True,
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, transform=transforms.ToTensor()
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
    shuffle=False,
)

train_counter = Counter()
for images, labels in train_loader:
    train_counter.update(labels.tolist())

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\nTraining label distribution:")
sorted(train_counter.items())

print("\nTest label distribution:")
sorted(test_counter.items())

# 
# ### Performance baseline

# 
# - Especially for imbalanced datasets, it's pretty helpful to compute a performance baseline.
# - In classification contexts, a useful baseline is to compute the accuracy for a scenario where the model always predicts the majority class -- we want our model to be better than that!

#
# %load ../code_dataset/performance_baseline.py
majority_class = test_counter.most_common(1)[0]
print("Majority class:", majority_class[0])

baseline_acc = majority_class[1] / sum(test_counter.values())
print("Accuracy when always predicting the majority class:")
print(f"{baseline_acc:.2f} ({baseline_acc*100:.2f}%)")

# 
# ## A quick visual check

#
# %load ../code_dataset/plot_visual-check_basic.py
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torchvision


for images, labels in train_loader:
    break


plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training images")
plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(images[:64], padding=1, normalize=True), (1, 2, 0)
    )
)
plt.savefig('training.png')

# 
# ### Setting up a `DataModule`

# 
# - There are three main ways we can prepare the dataset for Lightning. We can
#   1. make the dataset part of the model;
#   2. set up the data loaders as usual and feed them to the fit method of a Lightning Trainer -- the Trainer is introduced in the following subsection;
#   3. create a LightningDataModule.
# - Here, we will use approach 3, which is the most organized approach. The `LightningDataModule` consists of several self-explanatory methods, as we can see below:

#
import os

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path="./"):
        super().__init__()
        self.data_path = data_path
        print(os.getpid(), "😎 call prepare data....")
        self.prepare_data()

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_path, download=True)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((70, 70)),
                transforms.RandomCrop((64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((70, 70)),
                transforms.CenterCrop((64, 64)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )
        self.train, self.valid = random_split(train, lengths=[45000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        return test_loader




def train_me():
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="valid_acc")  # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name="mv3-cifar10", version="fuheng")

    import time

    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        deterministic=False,
        log_every_n_steps=10,
    )

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)

    runtime = (time.time() - start_time) / 60
    print(f"Training took {runtime:.2f} min in total.")
    return trainer

if __name__ == "__main__":
    torch.manual_seed(1)
    data_module = DataModule(data_path="./data")

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger


    lightning_model = LightningModel(pytorch_model, learning_rate=LEARNING_RATE)
    trainer = train_me()

    print("----------------------- EVAL -------------------------")

    import pandas as pd
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "valid_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    df_metrics[["train_acc", "valid_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    #plt.show()
    plt.savefig('eval.png')


    print("----------------------- TEST -------------------------")
    trainer.test(model=lightning_model, datamodule=data_module, ckpt_path='best')

    path = trainer.checkpoint_callback.best_model_path
    print(os.getpid(), path)

    predicted_labels = []
    lightning_model = LightningModel.load_from_checkpoint(path, model=pytorch_model)
    lightning_model.to('cuda')
    lightning_model.eval()
    test_dataloader = data_module.test_dataloader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    for batch in test_dataloader:
        features, true_labels = batch
        features = features.to(device)
        true_labels = true_labels.to('cpu')
        with torch.no_grad():
            logits = lightning_model(features)
            logits = logits.to(device)

        predicted_labels = torch.argmax(logits, dim=1)

        acc(predicted_labels.to('cpu'), true_labels)
    predicted_labels[:5]

    test_acc = acc.to('cuda').compute()
    print(f'Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')

    print("----------------------- ERROR ANALYSIS -------------------------")
    class_dict = {0: 'airplane',
                  1: 'automobile',
                  2: 'bird',
                  3: 'cat',
                  4: 'deer',
                  5: 'dog',
                  6: 'frog',
                  7: 'horse',
                  8: 'ship',
                  9: 'truck'}

    import sys

    sys.path.append("../../pytorch_ipynb")

    from helper_plotting import show_examples


    show_examples(
        model=lightning_model, data_loader=test_dataloader, class_dict=class_dict
    )

    print("----------------------- CMATRIX -------------------------")
    from torchmetrics import ConfusionMatrix
    import matplotlib
    from mlxtend.plotting import plot_confusion_matrix
    cmat = ConfusionMatrix(task="multiclass", num_classes=len(class_dict)).to(device)
    for x, y in test_dataloader:
        with torch.no_grad():
            pred = lightning_model(x.to(device))
        cmat(pred, y.to(device))
    cmat_tensor = cmat.compute()
    cmat = cmat_tensor.cpu().numpy()

    fig, ax = plot_confusion_matrix(
        conf_mat=cmat,
        class_names=class_dict.values(),
        norm_colormap=matplotlib.colors.LogNorm()
    )

    #plt.show()
    plt.savefig('confusion_matrix.png')
