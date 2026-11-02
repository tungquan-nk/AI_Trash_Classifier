
"""Huấn luyện ResNet18 phân loại rác bằng transfer learning.

Script đọc dữ liệu đã chia thành train/validation, tinh chỉnh mô hình pretrained,
theo dõi Macro-F1 để chọn checkpoint tốt nhất và lưu lịch sử huấn luyện.
"""

from pathlib import Path
import json
import random
import csv
import copy

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Seed dùng để cố định các nguồn ngẫu nhiên nhằm giúp kết quả có thể tái lập.
SEED = 42

# Các siêu tham số dùng trong quá trình học.
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Số tiến trình tải dữ liệu và số epoch liên tiếp không cải thiện trước khi dừng.
NUM_WORKERS = 2
PATIENCE = 7

# BASE_DIR là thư mục gốc dự án (thư mục cha của src); dữ liệu, checkpoint và
# artifact được định vị tương đối với thư mục này để script chạy từ đâu cũng được.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "datasets" / "split_v2"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ARTIFACT_DIR = BASE_DIR / "artifacts"

BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet18_v2_best.pth"
CLASSES_PATH = ARTIFACT_DIR / "resnet18_v2_classes.json"
HISTORY_PATH = ARTIFACT_DIR / "resnet18_v2_history.csv"


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

# Khởi tạo seed cho Python, NumPy và PyTorch; thiết lập seed CUDA nếu có GPU.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Ưu tiên GPU để tăng tốc; nếu không có CUDA thì chạy trên CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("RESNET18 - WASTE CLASSIFICATION V2")
print("=" * 50)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 3. IMAGE TRANSFORMS
# ==========================================

# Dùng trọng số pretrained mặc định của ResNet18 và lấy chuẩn hóa đầu vào
# (mean/std) tương ứng với dữ liệu mà mô hình đã được huấn luyện.
weights = ResNet18_Weights.DEFAULT

mean = weights.transforms().mean
std = weights.transforms().std

# Tăng cường dữ liệu chỉ áp dụng cho tập train, giúp mô hình bớt phụ thuộc
# vào góc nhìn, vị trí và màu sắc cố định của ảnh.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Validation không dùng augmentation ngẫu nhiên để phép đo ổn định, nhất quán.
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ==========================================
# 4. LOAD DATASET
# ==========================================

# ImageFolder tự gán nhãn theo tên thư mục con; train và val cần cùng mapping.
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

if train_dataset.classes != val_dataset.classes:
    raise ValueError("Train và validation class mapping không khớp!")

# Danh sách tên lớp và số đầu ra cần thiết của tầng phân loại cuối.
classes = train_dataset.classes
num_classes = len(classes)

print("\nClasses:", classes)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# Tạo các thư mục đầu ra nếu chưa tồn tại và lưu mapping chỉ số-lớp để dùng
# khi suy luận sau này.
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. DATA LOADERS
# ==========================================

# DataLoader gom ảnh thành batch; chỉ xáo trộn train, còn validation giữ thứ tự.
# pin_memory có lợi khi chuyển batch từ RAM sang GPU.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)


# ==========================================
# 6. INITIALIZE RESNET18
# ==========================================

# Tải ResNet18 đã pretrained. Thay tầng fully-connected cuối để số logits
# đầu ra bằng số lớp của bộ dữ liệu hiện tại.
model = models.resnet18(weights=weights)

in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)


# ==========================================
# 7. WEIGHTED CROSS ENTROPY
# ==========================================

# Đếm ảnh mỗi lớp. Trọng số nghịch đảo tỷ lệ lớp giúp giảm thiên lệch khi
# số lượng ảnh giữa các lớp không cân bằng.
class_counts = torch.bincount(
    torch.tensor(train_dataset.targets),
    minlength=num_classes
).float()

class_weights = class_counts.sum() / (
    num_classes * class_counts
)

class_weights = class_weights.to(device)

print("\nClass distribution:")

for class_name, count, weight in zip(
    classes,
    class_counts.tolist(),
    class_weights.tolist()
):
    print(
        f"{class_name:12s}: "
        f"{int(count):5d} images | "
        f"weight: {weight:.4f}"
    )

criterion = nn.CrossEntropyLoss(
# Loss dùng trọng số lớp; nhãn truyền vào phải là chỉ số lớp dạng số nguyên.
    weight=class_weights
)


# ==========================================
# 8. OPTIMIZER AND SCHEDULER
# ==========================================

# AdamW cập nhật toàn bộ tham số; ReduceLROnPlateau giảm learning rate khi
# Macro-F1 validation không cải thiện trong một số epoch.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==========================================
# 9. TRAINING VARIABLES
# ==========================================

# Theo dõi checkpoint có Macro-F1 cao nhất, epoch đạt được và early stopping.
best_val_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0

history = []


# ==========================================
# 10. TRAINING LOOP
# ==========================================

# Mỗi vòng lặp gồm một lượt cập nhật trên train, một lượt đánh giá trên val,
# sau đó cập nhật scheduler, ghi lịch sử và lưu mô hình tốt nhất.
for epoch in range(EPOCHS):

    # --------------------------------------
    # TRAIN
    # --------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # train() bật các hành vi huấn luyện như cập nhật thống kê BatchNorm.
    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Xóa gradient cũ, tính logits và loss, lan truyền ngược rồi cập nhật
        # trọng số mô hình bằng optimizer.
        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # Cộng loss theo số ảnh để cuối epoch tính trung bình chính xác,
        # kể cả khi batch cuối có kích thước nhỏ hơn.
        batch_size = labels.size(0)

        train_loss_sum += loss.item() * batch_size

        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        train_total += batch_size

    # Chỉ số train tổng hợp trên toàn bộ ảnh trong epoch.
    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    # eval() cố định hành vi của các lớp như BatchNorm/Dropout; inference_mode
    # giảm chi phí bộ nhớ vì không cần tính gradient khi đánh giá.
    model.eval()

    val_loss_sum = 0.0

    all_predictions = []
    all_labels = []

    # Thu thập dự đoán và nhãn thực để tính các chỉ số trên toàn tập validation.
    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            val_loss_sum += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Macro-F1 lấy trung bình F1 các lớp như nhau; weighted-F1 tính theo số
    # mẫu từng lớp. Cả hai bổ sung góc nhìn ngoài accuracy.
    val_loss = val_loss_sum / len(val_dataset)

    val_acc = accuracy_score(
        all_labels,
        all_predictions
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    val_weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------

    # Scheduler theo dõi Macro-F1; lấy learning rate sau bước scheduler để log.
    scheduler.step(val_macro_f1)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print(
        f"\nEpoch [{epoch + 1:02d}/{EPOCHS}] "
        f"LR: {current_lr:.6f}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2%}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2%} | "
        f"Val Macro-F1: {val_macro_f1:.4f}"
    )

    print(
        f"Val Weighted-F1: {val_weighted_f1:.4f}"
    )


    # --------------------------------------
    # SAVE HISTORY
    # --------------------------------------

    # Lưu các chỉ số của epoch hiện tại; ghi lại toàn bộ lịch sử để file CSV
    # vẫn phản ánh tiến trình nếu script dừng giữa chừng.
    history.append({
        "epoch": epoch + 1,
        "learning_rate": current_lr,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
    })

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    # Chỉ lưu checkpoint khi Macro-F1 tăng; nếu không thì tăng bộ đếm dừng sớm.
    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"  -> Saved best model! "
            f"Macro-F1: {best_val_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  -> No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )


    # --------------------------------------
    # EARLY STOPPING
    # --------------------------------------

    # Dừng khi đã đủ PATIENCE epoch liên tiếp mà không đạt checkpoint tốt hơn.
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# ==========================================
# 11. FINISH
# ==========================================

# In tóm tắt kết quả cuối cùng và đường dẫn tới các tệp đã tạo.
print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print("Best epoch:", best_epoch)
print("Best validation Macro-F1:", round(best_val_f1, 4))

print("Best model:", BEST_MODEL_PATH)
print("Classes:", CLASSES_PATH)
print("History:", HISTORY_PATH)
