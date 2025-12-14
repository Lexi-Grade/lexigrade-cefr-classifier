import pandas as pd
import os
from config import settings
from sklearn.model_selection import train_test_split

INPUT_FILE = f"{settings.raw_path}/universal_cefr_en_raw.csv"
OUTPUT_DIR = settings.split_path
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
df.dropna(subset=["cefr"], inplace=True)
print("Total samples:", len(df))
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

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

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
