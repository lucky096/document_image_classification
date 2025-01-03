import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.tensorboard import SummaryWriter

# Custom Dataset Class with Preprocessing
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, feature_extractor=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.feature_extractor = feature_extractor or ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Preprocess image using the feature extractor
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label

# Define the LightningModule
class ViTClassifier(LightningModule):
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=10, lr=2e-5):
        super(ViTClassifier, self).__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ViT model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        self.train_writer = SummaryWriter("logs/train")
        self.val_writer = SummaryWriter("logs/val")

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self(pixel_values)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        self.train_writer.add_scalar("Loss/Train", loss.item(), self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self(pixel_values)
        loss = outputs.loss
        acc = (outputs.logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_writer.add_scalar("Loss/Validation", loss.item(), self.global_step)
        self.val_writer.add_scalar("Accuracy/Validation", acc.item(), self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_train_end(self):
        self.train_writer.close()
        self.val_writer.close()

# Create Data Loaders
def get_dataloaders(batch_size=32):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    
    train_dataset = CIFAR10Dataset(root="data", train=True, feature_extractor=feature_extractor)
    val_dataset = CIFAR10Dataset(root="data", train=False, feature_extractor=feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    return train_loader, val_loader

# Instantiate Dataloaders
batch_size = 64
train_loader, val_loader = get_dataloaders(batch_size=batch_size)

# Callbacks for Model Saving and Early Stopping
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    mode="max",
    filename="best-checkpoint-{epoch:02d}-{val_acc:.2f}",
    save_top_k=1,
    verbose=True
)

early_stopping_callback = EarlyStopping(
    monitor="val_acc",
    patience=3,
    mode="max",
    verbose=True
)

# Instantiate the ViTClassifier
model = ViTClassifier(num_classes=10, lr=2e-5)

# Trainer
trainer = Trainer(
    max_epochs=10,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[checkpoint_callback, early_stopping_callback],
    progress_bar_refresh_rate=20,
    log_every_n_steps=10
)

# Train the Model
trainer.fit(model, train_loader, val_loader)