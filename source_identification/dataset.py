import torch
import os
import xml.etree.ElementTree as ET
from datasets import Dataset
from pathlib import Path

def load_data(path_input, path_label):
  project_root = Path().parent.absolute()
  data_dir = os.path.join(project_root, 'data', 'test')

  path2_x = os.path.join(data_dir, path_input)
  path2_y = os.path.join(data_dir, path_label)

  pun_instances = {}
  locations = {}
  human_data = []
  all_data = []

  tree = ET.parse(path2_x)
  root = tree.getroot()

  for child in root:
    line = []
    idx = child.attrib["id"]
    for kid in child:
      line.append(kid.text)
    pun_instances[idx] = line

  with open(path2_y) as gold:
    lines = gold.readlines()
    for line in lines:
      token = line.strip().split("\t")
      sub_tokens = token[1].split("_")
      locations[token[0]] = sub_tokens[2]

  for idx in pun_instances.keys():
    sentence = " ".join(pun_instances[idx])
    pun_word = pun_instances[idx][int(locations[idx]) - 1]
    pun_location = int(locations[idx]) - 1
    labels = [0] * len(pun_instances[idx])
    labels[pun_location] = 1
    all_data.append({ "tokens": pun_instances[idx], "labels": labels })
    human_data.append({ "sentence": sentence, "pun_word": pun_word })

  print('[INFO] Data loaded successfully.')
  return all_data


class SemEvalDataSet():
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def load_and_train_test_split(self, path_input, path_label):
    data = load_data(path_input, path_label)
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    return dataset
  
  def tokenize_and_align(self, examples):
    label_all_tokens = True
    tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"labels"]):
      word_ids = tokenized_inputs.word_ids(batch_index=i)
      previous_word_idx = None
      label_ids = []
      for word_idx in word_ids:
        if word_idx is None:
          label_ids.append(-100)
        elif word_idx != previous_word_idx:
          label_ids.append(label[word_idx])
        else:
          label_ids.append(label[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx
      labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

  def tokenize(self, examples):
    tokenized_dataset = examples.map(self.tokenize_and_align, batched=True)
    return tokenized_dataset