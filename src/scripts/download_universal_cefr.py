from datasets import load_dataset
from config import settings
import pandas as pd
import os


def main(language: str):
    output_file = os.path.join(settings.raw_base_path, language,  f"universal_cefr_{language}_raw.csv")
    rows = []

    print(f"Loading Universal CEFR ({language}) from Hugging Face...")
    if language == "english":
        universal_cefr_datasets = settings.universal_cefr_english_datasets
    else:
        universal_cefr_datasets = settings.universal_cefr_spanish_datasets

    for universal_cefr_dataset in universal_cefr_datasets.split(','):
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

        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved raw dataset to {output_file}")

    with open(output_file, "r") as f:
        df = pd.read_csv(f)
        df['cefr'] = df['cefr'].replace('A1+', 'A1')
        df['cefr'] = df['cefr'].replace('A2+', 'A2')
        df['cefr'] = df['cefr'].replace('B1+', 'B1')
        df['cefr'] = df['cefr'].replace('B2+', 'B2')
        df.to_csv(output_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    main(language="spanish")