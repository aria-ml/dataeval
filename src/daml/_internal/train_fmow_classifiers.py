import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.dataloader
import yaml
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, models
from torchvision.transforms import v2
from tqdm import tqdm

TRAIN_DIR = Path("/mnt/nas_device_0/fmow_chips_inc5_demo/train")
VAL_DIR = Path("/mnt/nas_device_0/fmow_chips_inc5_demo/val")
IMG_SIZE = [224, 224]


def select_device(device="", batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == "cpu"
    cuda_available = torch.cuda.is_available()
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # force torch.cuda.is_available()=False
    elif device:  # non-cpu device requested
        assert cuda_available, f"CUDA unavailable, invalid device {device} requested"  # check availability
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = (device.split(",") if device else range(torch.cuda.device_count())) # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert (batch_size % n == 0), f"batch-size {batch_size} not multiple of GPU count {n}"
    # JC, run on the device specified, single GPU...not training on multiple GPUs...
    return torch.device(f'cuda:{device}' if cuda else "cpu")
    # return torch.device(f'cuda:{device}' if cuda else "cpu")


class DemoClassifier():
    """ Totally not slapdash container for all the various and sundry attributes and methods """
    def __init__(self, config: dict):
        config["backbone"] = config.get("backbone","alex")
        config["device"] = config.get("device", "cpu")
        config["img_size"] = config.get("img_size", IMG_SIZE)
        config["batch_size"] = config.get("batch_size", 1)
        config["epochs"] = config.get("epochs",1)
        config["lr"] = config.get("lr", 5e-4)
        config["results_dir"] = config.get("results_dir", "/mnt/nas_device_0/increment5_classifiers")
        config["train_datadir"] = config.get("train_datadir","/mnt/nas_device_0/fmow_chips_inc5_demo/train")
        config["eval_datadir"] = config.get("eval_datadir", "/mnt/nas_device_0/fmow_chips_inc5_demo/val")
        config["results_dir"] = f"{config['results_dir']} / {config['backbone']}"
        
        self.results_dir = Path(config["results_dir"])
        self.train_dir = Path(config["train_datadir"])
        self.eval_dir = Path(config["eval_datadir"])
        self.backbone= config["backbone"]
        self.img_size = config["img_size"]
        self.num_epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        num_train_classes = len(list(self.train_dir.iterdir()))
        num_eval_classes = len(list(self.eval_dir.iterdir()))
        assert num_train_classes == num_eval_classes
        self.num_classes = num_train_classes
        self.model = self.initialize_model()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.device = select_device(device=config["device"])
        self.model.to(self.device)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        with open(self.results_dir/"training_params.yaml","w") as f:
            yaml.safe_dump(config, f)


    def train(self):
        train_loader = self.load_dataset(training=True)
        eval_loader = self.load_dataset(training=False)
        best_loss = 1e6
        epoch_pbar = tqdm(range(self.num_epochs))
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            eval_loss, acc = self.run_eval(eval_loader, epoch)
            desc = f"loss: train={train_loss:.3f}, epoch val={eval_loss:.3f}, best val={best_loss:.3f}"
            desc += f"val accuracy={acc:.3f}"
            epoch_pbar.set_description(desc)
            if eval_loss < best_loss:
                best_loss = eval_loss
                model_file = self.results_dir/"classifier_epoch{epoch}.pt"
                torch.save(self.model.state_dict(), model_file)

    
    def run_eval(self, eval_loader, epoch):
        self.model.eval()
        eval_loss = 0.
        acc_metric = MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        acc_metric.to(self.device)
        eval_pbar = tqdm(eval_loader)
        with torch.no_grad():
            for val_img, val_labels in eval_pbar:
                val_img = val_img.to(self.device, memory_format=torch.channels_last)
                val_labels = val_labels.to(self.device)
                val_pred = self.model(val_img)
                acc_metric.update(val_pred, val_labels)
                vloss = self.loss_fn(val_pred, val_labels)
                eval_loss = 0.85*eval_loss + 0.15*vloss
                eval_pbar.set_description(f"Eval: Epoch {epoch}, loss {eval_loss:.3f}")
        return eval_loss, acc_metric.compute()


    def train_epoch(self, train_loader, epoch):
        self.model.train(True)
        ema_loss = 0.
        train_pbar = tqdm(train_loader)
        for imgs, labels in train_pbar:
            self.optimizer.zero_grad()
            imgs = imgs.to(self.device, memory_format=torch.channels_last)
            labels = labels.to(self.device)
            mem = torch.cuda.memory_reserved() / 1e9 if self.device!="cpu" else 0
            pred_class = self.model(imgs)
            loss = self.loss_fn(pred_class, labels)
            loss.backward()
            self.optimizer.step()
            ema_loss = 0.85 * ema_loss + 0.15*loss.item()
            train_pbar.set_description(f"Training: Epoch {epoch}, memory {mem:.3f} GB, loss {ema_loss:.3f}")
        return ema_loss


    def initialize_model(self):
        """
        Return model with backbone loaded from torchvision builtins. Does not load pretrained
        weights. The final classification layer is replaced with one that matches the number of 
        classes we train on. 
        """
        if "RESNEXT" in self.backbone.upper():
            net_name = "resnext50_32x4d"
        elif "ALEX" in self.backbone.upper():
            net_name = "alexnet"
        return models.get_model(net_name, num_classes=self.num_classes)        


    def load_dataset(self, training=None):
        if training is None:
            training = self.model.training
        if training:
            dataset_dir = self.train_dir
            p_flip = 0.5
            shuffle = True
            crop_xfrm = v2.RandomResizedCrop(size=self.img_size, antialias=True)
        else:
            dataset_dir = self.eval_dir
            shuffle = False
            p_flip = 0.0
            crop_xfrm = v2.CenterCrop(size=self.img_size)
        img_xformer = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            crop_xfrm,
            v2.RandomHorizontalFlip(p=p_flip),
            v2.RandomVerticalFlip(p=p_flip),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225,0.221,0.212])    
        ])
        dataset = datasets.ImageFolder(str(dataset_dir), transform=img_xformer)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            pin_memory=True)
        return dataloader


def main():
    config = {
        "batch_size": 512,
        "backbone": "alex",
        "lr": 5e-4,
        "epochs": 5,
        "device": "0",
        "results_dir":   "/mnt/nas_device_0/increment5_classifiers",
        "train_datadir": "/mnt/nas_device_0/fmow_chips_inc5_demo/train",
        "eval_datadir":  "/mnt/nas_device_0/fmow_chips_inc5_demo/val",
    }
    alex_trainer = DemoClassifier(config)
    alex_trainer.train()

    config["backbone"] = "resnext"
    resnext_trainer = DemoClassifier(config)
    resnext_trainer.train()


if __name__ == "__main__":
    main()