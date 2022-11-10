import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM
# from dataset import SemEvalDataSet, load_data
from sound_similarity.sound_similarity import find_similar_sounding_words

MODEL = 'roberta-large'

tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = RobertaForMaskedLM.from_pretrained(MODEL)

# data = load_data(
#   path_input='subtask3-heterographic-test.xml',
#   path_label='subtask3-heterographic-test.gold'
# )
# train_data = SemEvalDataSet(data)
train_data = []

def get_target_word_from_pun(masked_sent, cand_list):
  inputs = tokenizer(masked_sent, return_tensors="pt")
  loss_arr = []

  for cand in tqdm(cand_list):
    sent_option = masked_sent.replace("<mask>", cand)
    labels = tokenizer(sent_option, return_tensors="pt")["input_ids"][0][:inputs.input_ids.shape[1]]
    outputs = model(**inputs, labels=labels)
    loss_arr.append(outputs.loss.item())

  min_idx = torch.argmin(torch.tensor(loss_arr)).item()
  pred_word = cand_list[min_idx]
  pred_sent = masked_sent.replace('<mask>', pred_word)

  return pred_word, pred_sent

src_target_list = []

def run_inference():
  for train in train_data:
    sent = train['sentence']
    src = train['src']
    tgt_root = train['tgt_root']
    masked_sent = sent.replace(src, '<mask>')
    cand_list = find_similar_sounding_words(src)

    if cand_list != []:
      target_pred_pun, target_sent = get_target_word_from_pun(masked_sent, cand_list)
      print("Sentence: ", sent)
      print("Source word: ", src)
      print("Target root: ", tgt_root)
      print("Predicted target: ", target_pred_pun)
      print("Predicted sentence: ", target_sent)
      print("===========================================================")
      src_target_list.append({ "src": src, "target": tgt_root, "pred": target_pred_pun, "sent": sent })

  return src_target_list

def run_inference_pipeline(sentence, source):
  masked_sent = sentence.replace(source, '<mask>')
  cand_list = find_similar_sounding_words(source)
  if cand_list != []:
    target_pred_pun, target_sent = get_target_word_from_pun(masked_sent, cand_list)
    print("===========================================================")
    print("Sentence: ", sentence)
    print("Source word: ", source)
    print("Predicted target: ", target_pred_pun)
    print("Predicted sentence: ", target_sent)
    print("===========================================================")
    return [{ "source": source, "predicted": target_pred_pun, "sentence": sentence }]

