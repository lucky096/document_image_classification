import json
from pathlib import Path
from platform import processor
from tkinter import Image
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

def predict_document_image(image_path, model, processor):
    json_path = image_path.with_suffix(".json")
    with open(json_path, "r") as f:
        ocr_result = json.load(f)
    
    with Image.open(image_path).convert("RGB") as img:
        width, height = img.size
        width_scale, height_scale = 1000/width, 1000/height

        words, boxes = [], []
        for row in ocr_result:
            boxes.append([
                row["bounding_box"][0] * width_scale,
                row["bounding_box"][1] * height_scale,
                row["bounding_box"][2] * width_scale,
                row["bounding_box"][3] * height_scale
            ])
            words.append(row["word"])
        
        encoding = processor(img, words, boxes=boxes, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        output = model(**encoding)

        predicted_class = output.logits.argmax().item()
    return model.config.id2label[predicted_class]


if __name__ == "__main__":
    model = LayoutLMv3ForSequenceClassification.from_pretrained("document_image_sequence_classifier")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    test_images = list(Path("test_images").glob("*.jpg"))
    labels = []
    predictions = []
    for image_path in test_images:
        labels.append(image_path.parent.name)
        predictions.append(predict_document_image(image_path, model, processor))

    
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()