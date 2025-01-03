"""
Using LayoutLMV3 model from Microsoft
V1 did not use image data
V2 and V3 uses image data

training using pytorch lightning module
training for classification of documents also called sequence classification in this case 
since we're classifying sequences of images + text + layout (bounding boxes)
"""

from operator import call
from pathlib import Path
import json
from numpy import test
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import LayoutLMv3Processor, LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3ForSequenceClassification
from torchmetrics import Accuracy


DOCUMENT_CLASSES = sorted(list(map(
    lambda p: p.name,
    Path("images").glob("*")
)))


def scale_bboxes(bbox, wscale, hscale):
    return [
        bbox[0] * wscale,
        bbox[1] * hscale,
        bbox[2] * wscale,
        bbox[3] * hscale
    ]

class DocumentImageClassificationDataset(Dataset):
    def __init__(self, batch, processor):
        self.batch = batch
        self.processor = processor
        pass

    def __get_item__(self, idx):
        image_path = self.batch[idx]
        json_path = image_path.replace(".jpg", ".json")
        with open(json_path, "r") as f:
            ocr_result = json.load(f)
        
        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size
            width_scale, height_scale = 1000/width, 1000/height

            words, boxes = [], []
            for row in ocr_result:
                boxes.append(scale_bboxes(row["bounding_box"], width_scale, height_scale))
                words.append(row["word"])
            
            encoding = self.processor(img, words, boxes=boxes, max_leght=512, padding="max_length", truncation=True, return_tensors="pt")
        
        labels = DOCUMENT_CLASSES.index(image_path.parent.name)
        return dict(
            input_ids = encoding.input_ids.squeeze(),
            attention_mask = encoding.attention_mask.squeeze(),
            bbox = encoding.bbox.squeeze(dim=1),
            pixel_values = encoding.pixel_values.squeeze(dim=1),
            labels = torch.tensor(labels, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.batch)

class DocumentImageSequenceClassifier(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_classes)
        self.model.config.id2label = {id: label for id, label in enumerate(DOCUMENT_CLASSES)}
        self.model.config.label2id = {label: id for id, label in enumerate(DOCUMENT_CLASSES)}
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values, labels=labels)

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values, labels=labels)
        self.log("train_loss", output.loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy(output.logits, labels), on_step=True, on_epoch=True, prog_bar=True)
        return output.loss
    
    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.val_loss("val_loss", output.loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return output.loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-5)
    
    def on_train_end(self):
        pass


if __name__ == "__main__":

    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    processor = LayoutLMv3Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    image_paths = sorted(list(Path("/Users/luckysrivastava/Workspace/data/DocumentImages/").glob("**/*.jpg")))
    train_images, test_images = train_test_split(image_paths, test_size=.2)

    train_dataset = DocumentImageClassificationDataset(train_images, processor)
    test_dataset = DocumentImageClassificationDataset(test_images, processor)

    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = DocumentImageSequenceClassifier(num_classes=len(DOCUMENT_CLASSES))

    model_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{val_loss:.4f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min"
)
    trainer = Trainer(
        accelerator="mps",
        precision=16,
        devices=1,
        max_epochs=5,
        callbacks=[model_checkpoint],
    )

    trainer.fit(model, train_loader, test_loader)

    trained_model = DocumentImageSequenceClassifier.load_from_checkpoint(model_checkpoint.best_model_path,
                                                                         n_classes=len(DOCUMENT_CLASSES),
                                                                         local_files_only=True)
    
    trained_model.save_pretrained("document_image_sequence_classifier")