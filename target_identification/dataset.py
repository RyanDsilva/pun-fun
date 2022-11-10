import torch
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def load_data(path_input, path_label):
  project_root = Path().parent.absolute()
  data_dir = os.path.join(project_root, 'data', 'test')

  path3_x = os.path.join(data_dir, path_input)
  path3_y = os.path.join(data_dir, path_label)

  sent_dict = {}
  src_tgt_dict = {}
  all_data = []

  tree = ET.parse(path3_x)
  root = tree.getroot()

  for child in root:
    idx = child.attrib["id"]
    line = []
    for kid in child:
      line.append(kid.text)
    sent_dict[idx] = line

  with open(path3_y) as gold3:
    lines = gold3.readlines()
    for line in lines:
      token = line.strip().split("\t")
      _, idx, loc = token[0].split("_")
      src = token[1].split('%')[0]
      tgt = token[2].split('%')[0]
      src_tgt_dict['het_' + str(idx)] = (src, tgt, int(loc))

  for key, sent_list in sent_dict.items():
      all_data.append({"sentence": " ".join(sent_list), "src": sent_list[src_tgt_dict[key][2] - 1], "src_root": src_tgt_dict[key][0], "tgt_root": src_tgt_dict[key][1]})

  print('[INFO] Data loaded successfully.')
  return all_data


class SemEvalDataSet(torch.utils.data.Dataset):
  def __init__(self, data):
    self.dataset = data

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sample_item = self.dataset[idx]
    sample = { 'sentence' : sample_item['sentence'], 'src' : sample_item['src'], 'src_root' : sample_item['src_root'], 'tgt_root' : sample_item['tgt_root'] }
    return (sample)