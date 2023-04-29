import datasets


data_train = datasets.load_dataset("json", data_files="train_en.70.jsonl")

data_validation = datasets.load_dataset("json", data_files="validation.jsonl")

data_test = datasets.load_dataset("json", data_files="test.jsonl")

#print(data_test)

# Filtra il dataset per la lingua inglese
data_validation_en = data_validation.filter(lambda example: example['lang'] == 'en')

data_test_en = data_test.filter(lambda example: example['lang'] == 'en')


#print(data_test_en)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")


def tokenize_function(examples):
    return tokenizer(examples["sent1"], padding="max_length", truncation=True)


tokenized_data_train = data_train.map(tokenize_function, batched=True)

tokenized_data_test = data_test_en.map(tokenize_function, batched=True)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("prithivida/parrot_paraphraser_on_T5", num_labels=7)


from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_test_en,
    compute_metrics=compute_metrics,
    label_column_name="annot_score"
)
