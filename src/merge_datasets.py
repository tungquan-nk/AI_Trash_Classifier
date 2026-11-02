
from pathlib import Path
import shutil
import csv
import hashlib

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

TRASHNET_DIR = BASE_DIR / "datasets/raw/trashnet"
REALWASTE_DIR = BASE_DIR / "datasets/raw/realwaste"
OUTPUT_DIR = BASE_DIR / "datasets/merged"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Ánh xạ tên lớp RealWaste về 6 lớp của dự án
REALWASTE_MAPPING = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash",
    "miscellaneous": "trash",
    "miscellaneous trash": "trash",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_hash(file_path):
    """Tạo hash để phát hiện ảnh trùng hoàn toàn."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_images(dataset_dir, mapping, source_name):
    """Thu thập ảnh và ánh xạ nhãn."""
    results = []

    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name.lower().strip()
        label = mapping.get(folder_name)

        if label is None:
            continue

        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            results.append({
                "path": image_path,
                "label": label,
                "source": source_name,
            })

    return results


def main():
    if not TRASHNET_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy: {TRASHNET_DIR}")

    if not REALWASTE_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy: {REALWASTE_DIR}")

    # Xóa thư mục merge cũ để tránh ảnh bị nhân đôi khi chạy lại
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for label in CLASSES:
        (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

    trashnet_mapping = {label: label for label in CLASSES}

    trashnet_images = collect_images(
        TRASHNET_DIR,
        trashnet_mapping,
        "trashnet"
    )

    realwaste_images = collect_images(
        REALWASTE_DIR,
        REALWASTE_MAPPING,
        "realwaste"
    )

    all_images = trashnet_images + realwaste_images

    seen_hashes = set()
    manifest = []
    counts = {}

    for item in all_images:
        source_path = item["path"]
        label = item["label"]
        source = item["source"]

        try:
            image_hash = get_hash(source_path)
        except OSError:
            print(f"Không đọc được: {source_path}")
            continue

        if image_hash in seen_hashes:
            print(f"Bỏ ảnh trùng: {source_path}")
            continue

        seen_hashes.add(image_hash)

        counts[label] = counts.get(label, 0) + 1
        filename = f"{source}_{label}_{counts[label]:05d}{source_path.suffix.lower()}"

        destination = OUTPUT_DIR / label / filename
        shutil.copy2(source_path, destination)

        manifest.append({
            "filename": filename,
            "label": label,
            "source": source,
            "original_path": str(source_path),
            "merged_path": str(destination),
            "sha256": image_hash,
        })

    # Lưu thông tin nguồn gốc ảnh
    manifest_path = OUTPUT_DIR / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label",
                "source",
                "original_path",
                "merged_path",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\n===== KẾT QUẢ MERGE =====")

    for label in CLASSES:
        count = len(list((OUTPUT_DIR / label).glob("*")))
        print(f"{label}: {count} ảnh")

    print(f"\nTổng số ảnh: {len(manifest)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
