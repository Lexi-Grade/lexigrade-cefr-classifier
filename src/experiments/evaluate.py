import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from config import settings



MODEL_DIR = f"{settings.models_path}/cefr_xlm_roberta_v1"
TEST_FILE = f"{settings.split_path}/test.csv"
MAX_LENGTH = 128
BATCH_SIZE = 16

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading test data...")
df = pd.read_csv(TEST_FILE)
df["label"] = df["cefr"].map(label2id)

test_ds = Dataset.from_pandas(df[["text", "label"]])

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

test_ds = test_ds.map(tokenize, batched=True)
test_ds.set_format("torch")


print("Running inference...")
preds = []
labels = []

with torch.no_grad():
    for i in range(0, len(test_ds), BATCH_SIZE):
        batch = test_ds[i:i+BATCH_SIZE]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

        preds.extend(batch_preds)
        labels.extend(batch["label"].numpy())


acc = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")

print("\n=== TEST RESULTS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")


cm = confusion_matrix(labels, preds, normalize="true")
print("\n=== CONFUSION MATRIX (normalized) ===")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        labels,
        preds,
        target_names=LABELS,
        digits=3
    )
)
