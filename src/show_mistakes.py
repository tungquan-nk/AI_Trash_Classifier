
from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "datasets" / "split" / "test"
MODEL_PATH = BASE_DIR / "checkpoints" / "resnet18_best.pth"
CLASSES_PATH = BASE_DIR / "artifacts" / "resnet18_classes.json"

OUTPUT_JSON = BASE_DIR / "artifacts" / "resnet18_mistakes.json"

BATCH_SIZE = 16
NUM_WORKERS = 2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET18 - PHÂN TÍCH ẢNH DỰ ĐOÁN SAI")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. KIỂM TRA FILE ĐẦU VÀO
# ==========================================
for path in [TEST_DIR, MODEL_PATH, CLASSES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")


# ==========================================
# 3. ĐỌC DANH SÁCH CÁC LỚP
# ==========================================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

print("Classes:", classes)


# ==========================================
# 4. TIỀN XỬ LÝ ẢNH
# ==========================================
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


# ==========================================
# 5. TẢI DATASET TEST
# ==========================================
test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=preprocess,
)

if test_dataset.classes != classes:
    raise ValueError(
        "Thứ tự lớp không khớp!\n"
        f"Trong file classes: {classes}\n"
        f"Trong dataset: {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

print("Tổng ảnh test:", len(test_dataset))


# ==========================================
# 6. KHỞI TẠO VÀ TẢI RESNET18
# ==========================================
model = models.resnet18(weights=None)

# Thay lớp phân loại cuối bằng số lớp rác
model.fc = nn.Linear(
    model.fc.in_features,
    len(classes),
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Đã tải checkpoint:", MODEL_PATH)


# ==========================================
# 7. DỰ ĐOÁN VÀ THU THẬP ẢNH SAI
# ==========================================
mistakes = []
pair_counts = Counter()
sample_index = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        # Chuyển kết quả về CPU để xử lý
        predictions = predictions.cpu().tolist()
        confidences = confidences.cpu().tolist()
        true_labels = labels.tolist()

        for i in range(len(true_labels)):
            true_id = true_labels[i]
            pred_id = predictions[i]

            # Lấy chính xác đường dẫn ảnh theo vị trí trong dataset
            image_path = test_dataset.samples[sample_index][0]

            if true_id != pred_id:
                item = {
                    "file": str(image_path),
                    "true": classes[true_id],
                    "pred": classes[pred_id],
                    "confidence": round(
                        float(confidences[i]), 4
                    ),
                }

                mistakes.append(item)

                pair_counts[
                    (classes[true_id], classes[pred_id])
                ] += 1

            sample_index += 1


# ==========================================
# 8. IN THỐNG KÊ
# ==========================================
total_images = len(test_dataset)
total_mistakes = len(mistakes)
correct_images = total_images - total_mistakes

accuracy = (
    correct_images / total_images
    if total_images > 0
    else 0.0
)

print("\n" + "=" * 60)
print("KẾT QUẢ")
print("=" * 60)

print("Tổng ảnh test:", total_images)
print("Dự đoán đúng:", correct_images)
print("Dự đoán sai:", total_mistakes)
print(f"Accuracy: {accuracy:.2%}")


# ==========================================
# 9. CÁC CẶP LỚP BỊ NHẦM NHIỀU NHẤT
# ==========================================
print("\n" + "=" * 60)
print("TOP CÁC CẶP LỚP BỊ NHẦM")
print("=" * 60)

if pair_counts:
    for (true_class, pred_class), count in pair_counts.most_common():
        print(
            f"{true_class:12s} -> "
            f"{pred_class:12s}: {count} ảnh"
        )
else:
    print("Không có ảnh nào bị dự đoán sai.")


# ==========================================
# 10. IN CHI TIẾT CÁC ẢNH SAI
# ==========================================
print("\n" + "=" * 60)
print("CHI TIẾT 20 ẢNH SAI ĐẦU TIÊN")
print("=" * 60)

for index, item in enumerate(mistakes[:20], start=1):
    print(f"\n{index}. Ảnh: {item['file']}")
    print(f"   Thực tế : {item['true']}")
    print(f"   Dự đoán : {item['pred']}")
    print(f"   Confidence: {item['confidence']:.2%}")


# ==========================================
# 11. LƯU KẾT QUẢ VÀO JSON
# ==========================================
OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": "ResNet18",
            "test_images": total_images,
            "correct": correct_images,
            "incorrect": total_mistakes,
            "accuracy": round(accuracy, 4),
            "mistakes": mistakes,
            "confusion_pairs": [
                {
                    "true": true_class,
                    "predicted": pred_class,
                    "count": count,
                }
                for (true_class, pred_class), count
                in pair_counts.most_common()
            ],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n" + "=" * 60)
print("ĐÃ HOÀN THÀNH")
print("=" * 60)
print("File kết quả:", OUTPUT_JSON)
