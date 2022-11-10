from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from dataset import SemEvalDataSet

MODEL = "roberta-large"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
DECAY_RATE = 0.01

tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)

semEvalDataset = SemEvalDataSet(tokenizer)
data = semEvalDataset.load_and_train_test_split(
  path_input='subtask2-heterographic-test.xml',
  path_label='subtask2-heterographic-test.gold'
)
tokenized_dataset = semEvalDataset.tokenize(data)

model_name = MODEL.split("/")[-1]
args = TrainingArguments(
  f"{model_name}-finetuned-semeval",
  evaluation_strategy = "epoch",
  learning_rate=LEARNING_RATE,
  per_device_train_batch_size=BATCH_SIZE,
  per_device_eval_batch_size=BATCH_SIZE,
  num_train_epochs=EPOCHS,
  weight_decay=DECAY_RATE,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
  model,
  args,
  train_dataset=tokenized_dataset["train"],
  eval_dataset=tokenized_dataset["test"],
  data_collator=data_collator,
  tokenizer=tokenizer,
)
trainer.train()
trainer.evaluate()
trainer.save_model('models')