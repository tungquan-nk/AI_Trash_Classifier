
from pathlib import Path
from collections import Counter
from PIL import Image

# Đường dẫn thư mục
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "raw" / "dataset-resized"
)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}\n"
        "Hãy sửa DATA_DIR theo vị trí thực tế của dataset."
    )

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
class_counts = Counter()
bad_images = []
image_sizes = Counter()

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    count = 0

    for image_path in class_dir.rglob("*"):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in extensions:
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image_sizes[image.size] += 1

            count += 1

        except Exception as error:
            bad_images.append((str(image_path), str(error)))

    class_counts[class_dir.name] = count

print("\n=== THỐNG KÊ DATASET ===")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name:12s}: {count:4d} ảnh")

print("Tổng số ảnh:", sum(class_counts.values()))
print("Số kích thước ảnh khác nhau:", len(image_sizes))
print("Số ảnh lỗi:", len(bad_images))

if bad_images:
    print("\nVí dụ ảnh lỗi:")
    for path, error in bad_images[:10]:
        print(path, "->", error)

if len(class_counts) != 6:
    print(
        "\nCẢNH BÁO: Số lớp không bằng 6. "
        "Hãy kiểm tra lại thư mục dữ liệu."
    )
