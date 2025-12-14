import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from config import settings


MODEL_NAME = "xlm-roberta-base"
SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

def main(language: str):
    if language == "english":
        labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    else:
        labels = ["A1", "A2", "B1", "B2", "C1"]
    num_labels = len(labels)
    train_file = os.path.join(settings.split_base_path, language, "train.csv")
    val_file = os.path.join(settings.split_base_path, language, "val.csv")
    output_dir = os.path.join(settings.models_base_path, language, f"lexigrade_{language}_cefr_classifier_v1")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}


    print("Loading data...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    train_df["label"] = train_df["cefr"].map(label2id)
    val_df["label"] = val_df["cefr"].map(label2id)


    label_counts = Counter(train_df["label"])
    total = sum(label_counts.values())

    class_weights = {
        label: total / count
        for label, count in label_counts.items()
    }

    weights = torch.tensor(
        [class_weights[i] for i in range(num_labels)],
        dtype=torch.float
    )


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        texts = [str(t) for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    train_ds = Dataset.from_pandas(
        train_df[["text", "label"]],
        preserve_index=False
    )

    val_ds = Dataset.from_pandas(
        val_df[["text", "label"]],
        preserve_index=False
    )
    #train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    #val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(pred):
        logits = pred.predictions
        labels = pred.label_ids

        preds = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro")
        }

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None
        ):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss


    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=SEED,
        report_to="none"
    )


    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training completed.")


if __name__ == "__main__":
    main(language="spanish")
