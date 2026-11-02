from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt


# 1. Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "datasets" / "split" / "test"
MODEL_PATH = BASE_DIR / "checkpoints" / "resnet18_best.pth"
CLASSES_PATH = BASE_DIR / "artifacts" / "resnet18_classes.json"
OUTPUT_DIR = BASE_DIR / "artifacts"

BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 2. Đọc danh sách lớp
with open(CLASSES_PATH, encoding="utf-8") as f:
    classes = json.load(f)

print("Classes:", classes)


# 3. Tiền xử lý giống validation
weights = ResNet18_Weights.DEFAULT
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])


# 4. Đọc tập test
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=preprocess,
)

if test_dataset.classes != classes:
    raise ValueError(
        f"Class mapping không khớp!\n"
        f"Checkpoint: {classes}\n"
        f"Test dataset: {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

print("Test images:", len(test_dataset))


# 5. Khởi tạo mô hình và tải checkpoint
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()


# 6. Dự đoán trên tập test
y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_pred.extend(predictions)
        y_true.extend(labels.tolist())


# 7. In các chỉ số đánh giá
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

print("\n========== KẾT QUẢ TEST ==========")
print(f"Accuracy : {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro-F1 : {macro_f1:.4f}")

print("\n========== BÁO CÁO THEO LỚP ==========")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0,
))


# 8. Vẽ và lưu confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes,
)

fig, ax = plt.subplots(figsize=(9, 8))
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False,
)
ax.set_title("ResNet18 - Test Confusion Matrix")
fig.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
matrix_path = OUTPUT_DIR / "resnet18_confusion_matrix.png"
fig.savefig(matrix_path, dpi=160)
plt.close(fig)

print("Confusion matrix:", matrix_path)
