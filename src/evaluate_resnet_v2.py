
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==========================================
# 1. CONFIG
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "datasets/split_v2/test"
MANIFEST_PATH = BASE_DIR / "datasets/split_v2/test_manifest.csv"

MODEL_PATH = BASE_DIR / "checkpoints/resnet18_v2_best.pth"
CLASSES_PATH = BASE_DIR / "artifacts/resnet18_v2_classes.json"

ARTIFACT_DIR = BASE_DIR / "artifacts"

BATCH_SIZE = 16
NUM_WORKERS = 2

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. LOAD CLASSES
# ==========================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

num_classes = len(classes)

print("Classes:", classes)


# ==========================================
# 3. PREPROCESSING
# ==========================================

weights = ResNet18_Weights.DEFAULT

mean = weights.transforms().mean
std = weights.transforms().std

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

if test_dataset.classes != classes:
    raise ValueError(
        f"Class mapping không khớp!\n"
        f"Model: {classes}\n"
        f"Test: {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

print("Test images:", len(test_dataset))


# ==========================================
# 5. LOAD MODEL
# ==========================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# ==========================================
# 6. INFERENCE
# ==========================================

all_predictions = []
all_labels = []

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ==========================================
# 7. OVERALL METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 55)
print("RESNET18 V2 - TEST RESULTS")
print("=" * 55)

print(f"Test images: {len(test_dataset)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

plt.figure(figsize=(9, 7))

plt.imshow(cm, interpolation="nearest")
plt.title("ResNet18 V2 - Confusion Matrix")
plt.colorbar()

positions = np.arange(num_classes)

plt.xticks(positions, classes, rotation=45)
plt.yticks(positions, classes)

threshold = cm.max() / 2

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.ylabel("Actual label")
plt.xlabel("Predicted label")

plt.tight_layout()

cm_path = ARTIFACT_DIR / "resnet18_v2_confusion_matrix.png"

plt.savefig(cm_path, dpi=200)
plt.close()

print("\nConfusion matrix:", cm_path)


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "image": [
        Path(path).name
        for path, _ in test_dataset.samples
    ],
    "true_label": [
        classes[label]
        for label in all_labels
    ],
    "predicted_label": [
        classes[pred]
        for pred in all_predictions
    ],
    "correct": [
        true == pred
        for true, pred in zip(
            all_labels,
            all_predictions
        )
    ]
})

# Ghép nguồn dữ liệu từ test manifest
if MANIFEST_PATH.exists():

    manifest = pd.read_csv(MANIFEST_PATH)

    predictions_df = predictions_df.merge(
        manifest[["filename", "source"]],
        left_on="image",
        right_on="filename",
        how="left"
    )

    predictions_df.drop(
        columns=["filename"],
        inplace=True
    )

predictions_path = ARTIFACT_DIR / "resnet18_v2_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("Predictions:", predictions_path)


# ==========================================
# 10. EVALUATION BY SOURCE
# ==========================================

if "source" in predictions_df.columns:

    print("\n===== RESULTS BY DATA SOURCE =====")

    for source, group in predictions_df.groupby("source"):

        source_accuracy = group["correct"].mean()

        source_macro_f1 = f1_score(
            group["true_label"],
            group["predicted_label"],
            labels=classes,
            average="macro",
            zero_division=0
        )

        print(f"\nSource: {source}")
        print("Images:", len(group))
        print(f"Accuracy: {source_accuracy:.4f}")
        print(f"Macro-F1: {source_macro_f1:.4f}")


print("\nEvaluation completed!")
