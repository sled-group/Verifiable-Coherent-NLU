import torch
from torch.utils.data import TensorDataset
from www.utils import get_sublist
import spacy
from www.dataset.ann import human_atts, att_to_idx, att_types
import progressbar
from time import sleep
from copy import deepcopy
import numpy as np

# Creates tensor dataset from featurized dataset
def get_tensor_dataset(dataset, label_key='plausible', add_spans=False, add_segment_ids=False, masked_lm_params=None):
  # print(len(dataset[0]['input_ids']))
  # print(len(dataset[1]['input_ids']))
  all_input_ids = torch.tensor([ex['input_ids'] for ex in dataset])
  all_input_mask = torch.tensor([ex['input_mask'] for ex in dataset])
  all_label_ids = torch.tensor([ex[label_key] for ex in dataset], dtype=torch.long)

  # Randomly mask half of tokens with label not -100
  if masked_lm_params is not None:
    mask_token_id = masked_lm_params['mask_token_id']
    mask_prob = masked_lm_params['mask_prob']
    token_mask = torch.randn(all_label_ids.shape)
    if mask_token_id is not None:
      all_input_ids = all_input_ids.clone()
      all_input_ids[(all_label_ids != -100) & (token_mask <= mask_prob)] = mask_token_id
    else:
      all_input_mask = all_input_mask.clone()
      all_input_ids[(all_label_ids != -100) & (token_mask <= mask_prob)] = masked_lm_params['unk_token_id']
      # all_input_mask[(all_label_ids != -100) & (token_mask <= mask_prob)] = 0

  if add_spans:
    all_spans = torch.tensor([[ex['entity_span'][0]] + [ex['entity_span'][-1]] if len(ex['entity_span']) > 0 else [-1, -1] for ex in dataset], dtype=torch.long)
    tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_spans)
  elif add_segment_ids and 'segment_ids' in dataset[0]:
    all_segment_ids = torch.tensor([ex['segment_ids'] for ex in dataset])
    tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_segment_ids)
  else:
    tensor_dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
  return tensor_dataset

# Creates tensor dataset from featurized dataset
def get_tensor_dataset_tiered(dataset, max_sentences, add_segment_ids=False):
  # All inputs should be shaped something like (# stories, # )
  max_entities = max([len(story['entities']) for ex_2s in dataset for story in ex_2s['stories']])
  # max_sentences = max([len(ex['sentences']) for ex_2s in dataset for ex in ex_2s['stories']])
  seq_length = len(dataset[0]['stories'][0]['entities'][0]['input_ids'][0])
  num_attributes = len(dataset[0]['stories'][0]['entities'][0]['preconditions'][0])
  num_spans = len(dataset[0]['stories'][0]['entities'][0]['span_labels'])
  # print(max_entities, max_sentences, seq_length, num_attributes, num_spans)

  all_input_ids = torch.tensor([[[[story['entities'][e]['input_ids'][s] if e < len(story['entities']) else np.zeros((seq_length)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
  all_lengths = torch.tensor([[[len(story['sentences']) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
  num_entities = torch.tensor([[len(story['entities']) for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.int64)
  all_input_mask = torch.tensor([[[[story['entities'][e]['input_mask'][s] if e < len(story['entities']) else np.zeros((seq_length)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
  all_attributes = torch.tensor([[[[story['entities'][e]['attributes'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  all_preconditions = torch.tensor([[[[story['entities'][e]['preconditions'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  all_effects = torch.tensor([[[[story['entities'][e]['effects'][s] if e < len(story['entities']) and s < len(story['entities'][0]) else np.zeros((num_attributes)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  # all_spans = torch.tensor([[[story['entities'][e]['conflict_span'] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  # all_spans = torch.tensor([[[story['entities'][e]['span_labels'] if e < len(story['entities']) else np.zeros((num_spans)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  all_spans = torch.tensor([[[story['entities'][e]['conflict_span_onehot'] if e < len(story['entities']) else np.zeros((max_sentences)) for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset], dtype=torch.long)
  all_label_ids = torch.tensor([ex['label'] for ex in dataset], dtype=torch.long)

  # print(all_input_ids.shape)
  # print(all_lengths.shape)
  # print(all_input_mask.shape)
  # print(all_attributes.shape)
  # print(all_preconditions.shape)
  # print(all_effects.shape)
  # print(all_spans.shape)
  # print(all_label_ids.shape)

  if add_segment_ids and 'segment_ids' in dataset[0]:
    all_input_ids = torch.tensor([[[[story['entities'][e]['segment_ids'][s] if e < len(story['entities']) else np.zeros((seq_length)) for s in range(max_sentences)] for e in range(max_entities)] for story in ex_2s['stories']] for ex_2s in dataset])
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids, all_segment_ids)
  else:
    tensor_dataset = TensorDataset(all_input_ids, all_lengths, num_entities, all_input_mask, all_attributes, all_preconditions, all_effects, all_spans, all_label_ids)
  return tensor_dataset

# Tokenize, numericalize, and pad an input dataset
def add_bert_features_ConvEnt(dataset, tokenizer, seq_length, add_segment_ids=False):
  for i, ex in enumerate(dataset):
    exid = ex['id']
    ex['example_id'] = exid

    # Numericalize the data
    dialog = ' '.join(['Speaker%s: %s' % (turn['speaker'], turn['text']) for turn in ex['turns']])
    inputs = tokenizer.encode_plus(dialog,
                                    text_pair=ex['hypothesis'],
                                    add_special_tokens=True,
                                    max_length=seq_length,
                                    truncation=True)
    input_ids = inputs['input_ids']
    if 'token_type_ids' in inputs:
      token_type_ids = inputs['token_type_ids']

    # Don't want to truncate any data
    assert not('num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0)
    assert len(input_ids) <= seq_length

    # Pad to sequence length of 128
    padding_length = seq_length - len(input_ids)
    input_length = len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = [1] * input_length + ([0] * padding_length) # Mask will zero out padding tokens
    if add_segment_ids and 'token_type_ids' in inputs:
      token_type_ids = token_type_ids + [0] * padding_length

    assert len(input_ids) == len(attention_mask) == seq_length

    dataset[i]['input_ids'] = input_ids
    dataset[i]['input_mask'] = attention_mask
    if add_segment_ids and 'token_type_ids' in inputs:
      dataset[i]['segment_ids'] = token_type_ids

  return dataset

# Tokenize, numericalize, and pad an input two-story dataset
def add_bert_features_art(dataset, tokenizer, seq_length, add_segment_ids=False):
  for i, ex in enumerate(dataset):
    if 'input_ids' in dataset[i]:
      del dataset[i]['input_ids']
    if 'input_mask' in dataset[i]:
      del dataset[i]['input_mask']
    if 'segment_ids' in dataset[i]:
      del dataset[i]['segment_ids']

    exid = ex['id']
    ex['example_id'] = exid

    for story in ['1', '2']:
      # Numericalize the data
      story_text = ex['observation_1'] + ' ' + ex['hypothesis_%s' % story] + ' ' + ex['observation_2']
      inputs = tokenizer.encode_plus(story_text,
                                    add_special_tokens = True,
                                    max_length=seq_length,
                                    truncation=True)
      input_ids = inputs['input_ids']
      if add_segment_ids and 'token_type_ids' in inputs:
        token_type_ids = inputs['token_type_ids']
      else:
        token_type_ids = None

      # Don't want to truncate any data
      assert not('num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0)
      assert len(input_ids) <= seq_length

      # Pad to sequence length of 128
      padding_length = seq_length - len(input_ids)
      input_length = len(input_ids)
      input_ids = input_ids + ([0] * padding_length)
      attention_mask = [1] * input_length + ([0] * padding_length) # Mask will zero out padding tokens
      if token_type_ids is not None:
        token_type_ids = token_type_ids + [0] * padding_length

      assert len(input_ids) == len(attention_mask) == seq_length

      if 'input_ids' not in dataset[i]:
        dataset[i]['input_ids'] = []
      dataset[i]['input_ids'].append(input_ids)

      if 'input_mask' not in dataset[i]:
        dataset[i]['input_mask'] = []
      dataset[i]['input_mask'].append(attention_mask)

      if token_type_ids is not None:
        if 'segment_ids' not in dataset[i]:
          dataset[i]['segment_ids'] = []
        dataset[i]['segment_ids'].append(token_type_ids)

  return deepcopy(dataset)

# Tokenize, numericalize, and pad input dataset for tiered "reasoning"
def add_bert_features_tiered(dataset, tokenizer, seq_length, add_segment_ids=False):
  nlp = spacy.load("en_core_web_sm")
  # print(len(dataset['train']))
  # print(len(dataset['train'][0]['stories'])) 
  # print(len(dataset['train'][0]['stories'][0]['sentences'])) 
  max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
  for p in dataset:
    bar_size = len(dataset[p])
    bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar_idx = 0
    bar.start()
    for i, ex_2s in enumerate(dataset[p]):
      for s_idx, ex_1s in enumerate(ex_2s['stories']):
        for ent_idx, ex in enumerate(ex_1s['entities']):
          exid = ex['example_id']

          # Generate inputs for each sentence
          all_input_ids = np.zeros((max_story_length, seq_length))
          all_input_mask = np.zeros((max_story_length, seq_length))
          if add_segment_ids:
            all_segment_ids = np.zeros((max_story_length, seq_length))

          for j, sent in enumerate(ex['sentences']):
            inputs = tokenizer.encode_plus(ex['entity'], 
                                          text_pair=sent, 
                                          add_special_tokens=True, 
                                          max_length=seq_length, 
                                          truncation=True)
            input_ids = inputs['input_ids']

            if add_segment_ids and 'token_type_ids' in inputs:
              token_type_ids = inputs['token_type_ids']
            else:
              token_type_ids = None

            # Don't want to truncate any data
            assert not('num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0)
            assert len(input_ids) <= seq_length

            # Pad to sequence length of 128
            padding_length = seq_length - len(input_ids)
            input_length = len(input_ids)
            input_ids = input_ids + ([0] * padding_length)
            all_input_ids[j, :] = input_ids

            attention_mask = [1] * input_length + ([0] * padding_length) # Mask will zero out padding tokens
            all_input_mask[j, :] = attention_mask
            
            if token_type_ids is not None:
              token_type_ids = token_type_ids + [0] * padding_length
              all_segment_ids[j, :] = token_type_ids

            assert len(input_ids) == len(attention_mask) == seq_length
        
          dataset[p][i]['stories'][s_idx]['entities'][ent_idx]['input_ids'] = all_input_ids
          dataset[p][i]['stories'][s_idx]['entities'][ent_idx]['input_mask'] = all_input_mask
          if add_segment_ids and 'token_type_ids' in inputs:
            dataset[p][i]['stories'][s_idx]['entities'][ent_idx]['segment_ids'] = all_segment_ids

      bar_idx += 1
      bar.update(bar_idx)
    bar.finish()

  return dataset  
