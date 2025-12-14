from datasets import load_dataset
from config import settings
import pandas as pd
import os

OUTPUT_DIR = settings.raw_path
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "universal_cefr_en_raw.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
rows = []

print("Loading Universal CEFR (English) from Hugging Face...")
for universal_cefr_dataset in settings.universal_cefr_english_datasets.split(','):
    dataset = load_dataset(
        universal_cefr_dataset,
        split="train",
    )


    for split in dataset:
            text = split.get("text", "").strip()
            cefr = split.get("cefr_level", "").strip()

            if not text or not cefr:
                continue

            rows.append({
                "text": text,
                "cefr": cefr,
                "source_split": split
            })

    df = pd.DataFrame(rows)

    print(f"Total samples collected: {len(df)}")
    print(df["cefr"].value_counts())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved raw dataset to {OUTPUT_FILE}")

with open(OUTPUT_FILE, "r") as f:
    df = pd.read_csv(f)
    df['cefr'] = df['cefr'].replace('A1+', 'A1')
    df['cefr'] = df['cefr'].replace('A2+', 'A2')
    df['cefr'] = df['cefr'].replace('B1+', 'B1')
    df['cefr'] = df['cefr'].replace('B2+', 'B2')
    print(df["cefr"].value_counts())