import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



data_train_en = datasets.load_dataset("json", data_files="train_en.70.jsonl")

data_test = datasets.load_dataset("json", data_files="test.jsonl")
data_validation = datasets.load_dataset("json", data_files="validation.jsonl")

# Filtra il dataset per la lingua inglese
data_test_en = data_test.filter(lambda example: example['lang'] == 'en')
data_validation_en = data_validation.filter(lambda example: example['lang'] == 'en')


print(data_train_en)
print(data_test_en)
print(data_validation_en)

'''
# Trasformo la colonna annot_score nel tipo ClassLabels
data_train_en = data_train_en.map(lambda examples: {'label': ClassLabel(examples['quality'])}, remove_columns=['quality'])
data_test_en = data_test_en.map(lambda examples: {'label': ClassLabel(examples['annot_score'])}, remove_columns=['annot_score'])
data_validation_en = data_validation_en.map(lambda examples: {'label': ClassLabel(examples['annot_score'])}, remove_columns=['annot_score'])
'''

tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")


def tokenize_function(examples):
    return tokenizer(examples["sent1"], examples["sent2"], padding="max_length", truncation=True)

# Tokenize Dataset


tokenized_data_train = data_train_en.map(tokenize_function, batched=True)
tokenized_data_test = data_test_en.map(tokenize_function, batched=True)
tokenized_data_validation = data_validation_en.map(tokenize_function, batched=True)

# clean


def clean(dataset):

    dataset = dataset.remove_columns(["orig_id", "lang", "gem_id", "sent1", "sent2"])
    if dataset.has_column("annot_score"):
        dataset = dataset.rename_column("annot_score", "labels")
    elif dataset.has_column("quality"):
        dataset = dataset.rename_column("quality", "labels")

    dataset = dataset.with_format("torch")

    return dataset


tokenized_data_train = clean(tokenized_data_train)
tokenized_data_test = clean(tokenized_data_test)
tokenized_data_validation = clean(tokenized_data_validation)

print(tokenized_data_train)
print(tokenized_data_test)
print(tokenized_data_validation)

tokenized_data_train = tokenized_data_train["train"]
tokenized_data_test = tokenized_data_test["train"]
tokenized_data_validation = tokenized_data_validation["train"]


## Train with PyTorch Trainer
print("...TRAIN...")

model = AutoModelForSequenceClassification.from_pretrained("prithivida/parrot_paraphraser_on_T5", num_labels=7)

print("...training hyperparameters...")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

import numpy as np
import evaluate

print("...evaluate...")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


print("...trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    compute_metrics=compute_metrics,
)

print("!!! train !!!")

trainer.train()
