
from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = BASE_DIR / "datasets/merged/manifest.csv"
OUTPUT_DIR = BASE_DIR / "datasets/split_v2"

RANDOM_STATE = 42

def main():
    df = pd.read_csv(MANIFEST_PATH)

    # Phân tầng theo cả lớp và nguồn dataset
    df["stratify_key"] = df["label"] + "_" + df["source"]

    # Chia 70% train, 30% phần còn lại
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["stratify_key"],
    )

    # Chia phần còn lại thành 15% validation và 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["stratify_key"],
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # Tạo thư mục và copy ảnh
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            src = BASE_DIR / row["merged_path"].replace(
                str(BASE_DIR) + "/", ""
            )

            dst_dir = OUTPUT_DIR / split_name / row["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst_dir / row["filename"])

    # Lưu manifest cho từng tập
    for split_name, split_df in splits.items():
        split_df.drop(columns=["stratify_key"]).to_csv(
            OUTPUT_DIR / f"{split_name}_manifest.csv",
            index=False,
        )

    print("\n===== KẾT QUẢ CHIA DATASET =====")

    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}: {len(split_df)} ảnh")
        print(pd.crosstab(
            split_df["label"],
            split_df["source"]
        ))

    print("\nĐã lưu tại:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
