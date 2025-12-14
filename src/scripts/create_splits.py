import pandas as pd
import os
from config import settings
from sklearn.model_selection import train_test_split

SEED = 42

def main(language: str):
    input_file = os.path.join(settings.raw_base_path, language, f"universal_cefr_{language}_raw.csv")
    output_dir = os.path.join(settings.split_base_path, language)

    print("Loading dataset...")
    df = pd.read_csv(input_file)
    df.dropna(subset=["text", "cefr"], inplace=True)
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    df = df[df["text"].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    df = df.drop_duplicates(subset=["text"], keep="first")

    print("Total samples after cleaning:", len(df))
    print(df["cefr"].value_counts())

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["cefr"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["cefr"],
        random_state=SEED
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    print("\nTrain distribution:")
    print(train_df["cefr"].value_counts())

    print("\nVal distribution:")
    print(val_df["cefr"].value_counts())

    print("\nTest distribution:")
    print(test_df["cefr"].value_counts())

    print("\nSplits saved to data/splits/")

if __name__ == "__main__":
    main(language="spanish")
