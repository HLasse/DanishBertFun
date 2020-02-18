"""
This us trying to finetune the Danish BotXO BERT

oh and this is a proper docstring ;) 
"""
from simpletransformers.ner import NERModel

import torch
import os 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertForTokenClassification, BertForPreTraining, load_tf_weights_in_bert
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from danlp.datasets import DDT
import pyconll
import re
import unicodedata

######################################################
# Loading Data
######################################################


ddt = DDT()

conllu_format = ddt.load_as_conllu()
L = [(i, token.form, token.misc.get("name").pop()) for i, sent in enumerate(conllu_format) for token in sent]
df = pd.DataFrame(L, columns=['sentence_id', 'words', 'labels'])

######################################################
# to bert tokens 
######################################################
sent_str = [sent.text for sent in conllu_format]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sent_str]

# Convert tokens to indexes
with open("/home/au554730/Desktop/BERT_test/danish_bert_uncased/vocab.txt") as f:
    vocab = f.read()
vocab = vocab.split("\n")
vocab_d = {e: i for i, e in enumerate(vocab)}

def sentence_to_idx(sent):
    return [vocab_d.get(token, vocab_d["[UNK]"]) for token in sent]

input_ids = [sentence_to_idx(t) for t in tokenized_texts]

max_len = 128

# Padding
input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


######################################################
# loading model
######################################################

df['ner']

def load_BFTC_from_TF_ckpt(bert_config, ckpt_path, num_labels):
    """
    Helper function for loading model - workaround to prevent error
    """
    config = BertConfig.from_json_file(bert_config)
    model = BertForPreTraining(config)
    load_tf_weights_in_bert(model, ckpt_path)
    state_dict=model.state_dict()
    model = BertForTokenClassification(config, num_labels=num_labels)

    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
        start_prefix = 'bert.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           model.__class__.__name__, "\n\t".join(error_msgs)))
    return model

model = load_BFTC_from_TF_ckpt("danish_bert_uncased/config.json", "danish_bert_uncased/model.ckpt", num_labels = )

############################
#Implementing big and nasty D's
############################

# # for the big D-classifier ;)
# def start_d(sent):
#     return [token[0].lower() == "d"  for token in sent]
# true_class = [start_d(sent) for sent in tokenized_texts]

conllu_format = ddt.load_as_conllu()
L = [(i, token.form, token.misc.get("name").pop()) for i, sent in enumerate(conllu_format) for token in sent]
df = pd.DataFrame(L, columns=['sentence_id', 'words', 'labels'])

torch.save(model, "model.bin")
train_data = [
    [0, 'Simple', 'B-MISC'], [0, 'Transformers', 'I-MISC'], [0, 'started', 'O'], [1, 'with', 'O'], [0, 'text', 'O'], [0, 'classification', 'B-MISC'],
    [1, 'Simple', 'B-MISC'], [1, 'Transformers', 'I-MISC'], [1, 'can', 'O'], [1, 'now', 'O'], [1, 'perform', 'O'], [1, 'NER', 'B-MISC']
]
train_df = pd.DataFrame(train_data, columns=['sentence_id', 'words', 'labels'])

model = NERModel('bert', '/home/au554730/Desktop/BERT_test/danish_bert_pytorch/', use_cuda = False)

sub = df.head(100)

model.train_model(sub)


# Værsgo Kenneth
tokenized_texts = []
mylabels = []
for sent, tags in zip(sentences,labels):
BERT_texts = []
BERT_labels = np.array([])
for word, tag in zip(sent.split(),tags):
sub_words = tokenizer.wordpiece_tokenizer.tokenize(word)
tags = np.array([tag for x in sub_words])
tags[1:] = ‘X’
BERT_texts += sub_words
BERT_labels = np.append(BERT_labels,tags)
mytexts.append(BERT_texts)
mylabels.append(BERT_labels)