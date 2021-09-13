import argparse
import random
import os
import copy
from www.dataset.ann import att_to_idx, att_change_dir, att_types, att_default_values
from www.utils import print_dict
from collections import Counter
import numpy as np
from itertools import product
import math
import json
import jsonlines
import progressbar
from time import sleep

def split_list_by_ratio(list_in, ratios):
  len_list = len(list_in)
  assert len_list > 0
  assert 0.99<sum(ratios)<1.01
  random.shuffle(list_in)
  list_sp = []
  start = 0
  for i in range(len(ratios)):
    end = start + int(round(ratios[i]*len_list))
    if i == len(ratios) -1: # make sure all elements left goes to the last split
        end = len_list
    list_sp.append(list_in[start:end])
    start = end
  return list_sp

attributes = list(att_to_idx.keys())

# Balances class labels in a list of examples
def balance_labels(dataset, label_key='label'):
  label_counts = Counter([e[label_key] for e in dataset])
  examples_by_label = {}

  for l in label_counts:
    examples_by_label[l] = [e for e in dataset if e[label_key] == l]

  # Class 0/plausible class will always be dominating due to setup of data;
  # make each class the size of the most common POSITIVE label (second most common)
  max_count = label_counts.most_common(2)[1][1]

  new_dataset = []
  for l in label_counts:
    c = label_counts[l]

    if c < max_count:
      # If not enough examples, take all the examples we have, then randomly upsample some
      new_dataset += examples_by_label[l]
      new_dataset += list(np.random.choice(examples_by_label[l], size=max_count-c, replace=True))
    elif c > max_count:
      # If too many examples, randomly downsample
      new_dataset += list(np.random.choice(examples_by_label[l], size=max_count, replace=True))
    else:
      new_dataset += examples_by_label[l]

  return new_dataset

def convert_labels_to_dist(dataset, omit_unlabeled=False):
  if omit_unlabeled:
    dataset = [ex for ex in dataset if ex['label'] in {0, 1}]

  for ex in dataset:
    if ex['label'] == 0:
      ex['label'] = [1.0, 0.0]
    elif ex['label'] == 1:
      ex['label'] = [0.0, 1.0]
    else: # Some examples may have None or -1 label, which means unsure
      if not omit_unlabeled:
        ex['label'] = [0.5, 0.5]

  return dataset

def get_ConvEnt_spans(dataset):
  new_dataset = []
  for ex in dataset:
    num_turns = len(ex['turns'])
    for c1 in range(num_turns):
      for c2 in range(c1, num_turns):
        new_ex = ex.copy()
        new_ex['base_label'] = new_ex['label']
        new_ex['turns'] = new_ex['turns'][c1:c2+1]
        new_ex['example_id'] = str(new_ex['id']) + '-sp%s:%s' % (str(c1), str(c2))
        new_ex['id'] = new_ex['example_id']
        if ex['label'] == 1:
          if c1 <= new_ex['conflict_pair'][0] and c2 >= new_ex['conflict_pair'][1]:
            new_ex['label'] = 1
          else:
            new_ex['label'] = 0
            new_ex['conflict_pair'] = []
        else:
          new_ex['label'] = 0
          new_ex['conflict_pair'] = []
        new_dataset.append(new_ex)

  return new_dataset

def get_art_spans(dataset):
  new_dataset = []
  for ex in dataset:
    # possible spans: (1,3) (2,3) (1,2) (2,2)

    span_labels = {}
    cp = (ex['conflict_pair'][0], ex['conflict_pair'][1])
    if cp == (0, 2):
      span_labels[(0,2)] = ex['label']
      span_labels[(0,1)] = -1
      span_labels[(1,2)] = -1
      span_labels[(0,0)] = -1
      span_labels[(1,1)] = -1
      span_labels[(2,2)] = -1
    elif cp == (0, 1):
      span_labels[(0,2)] = ex['label']
      span_labels[(0,1)] = ex['label']
      span_labels[(1,2)] = ex['label'] if not ex['23_plausible'] else -1
      span_labels[(0,0)] = -1
      span_labels[(1,1)] = -1
      span_labels[(2,2)] = -1
    elif cp == (1, 2):
      span_labels[(0,2)] = ex['label']
      span_labels[(0,1)] = -1
      span_labels[(1,2)] = ex['label']
      span_labels[(0,0)] = -1
      span_labels[(1,1)] = -1
      span_labels[(2,2)] = -1
    elif cp == (1, 1):
      span_labels[(0,2)] = ex['label']
      span_labels[(0,1)] = ex['label']
      span_labels[(1,2)] = ex['label']
      span_labels[(0,0)] = -1
      span_labels[(1,1)] = ex['label']
      span_labels[(2,2)] = -1
    

    for t in [(0,1,2), (0,1), (1,2), (0,0), (1,1), (2,2)]:
      new_ex = ex.copy()
      new_ex['base_label'] = new_ex['label']
      new_ex['example_id'] = str(new_ex['id']) + '-sp%s:%s' % (str(min(t)), str(max(t)))
      new_ex['id'] = new_ex['example_id']
      if 0 not in t:
        new_ex['observation_1'] = ''
      if 1 not in t:
        new_ex['hypothesis_1'] = ''
        new_ex['hypothesis_2'] = ''
      if 2 not in t:
        new_ex['observation_2'] = ''

      new_ex['label'] = span_labels[(min(t), max(t))]
      if new_ex['label'] == -1:
        new_ex['conflict_pair'] = None

      new_dataset.append(new_ex)
  return new_dataset
      
# Generates a dataset for the tiered model for 2-story classification
def get_tiered_data(dataset):
  # states = {p: [] for p in dataset}
  max_story_length = max([len(ex['sentences']) for p in dataset for ex_2s in dataset[p] for ex in ex_2s['stories']])
  for p in dataset:
    for ex_2s in dataset[p]:
      for s_idx, ex in enumerate(ex_2s['stories']):
        if 'states' in ex:
          ent_sent_examples = {}
          all_entities = set()
          for i, sent_ann in enumerate(ex['states']):
            entities = []
            entity_anns = {}
            for att in sent_ann:
              for ent, v in [tuple(ann) for ann in sent_ann[att]]:
                entities.append(ent)
                if ent not in entity_anns:
                  entity_anns[ent] = [[0] * len(att_to_idx), [0] * len(att_to_idx)] # pre/post condition, then value for each attribute
                if 'location' not in att:
                  entity_anns[ent][0][att_to_idx[att]] = att_change_dir['default'][v][0] + 1
                  entity_anns[ent][1][att_to_idx[att]] = att_change_dir['default'][v][1] + 1
                else:
                  # For location, just use original label space - NOTE: missing this caused an issue for the "location" attribute in our original submission, but not the "h_location" attribute
                  entity_anns[ent][0][att_to_idx[att]] = v
                  entity_anns[ent][1][att_to_idx[att]] = v
            entities = list(set(entities))
            all_entities = all_entities.union(set(entities))

            for ent in entities:
              states_ex = {}
              states_ex['example_id'] = ex_2s['example_id'] + '-%s-%s-%s' % (str(s_idx), str(i), ent)
              states_ex['base_id'] = ex['example_id']
              states_ex['sentence_idx'] = i
              states_ex['entity'] = ent
              states_ex['sentence'] = ex['sentences'][i]
              states_ex['preconditions'] = entity_anns[ent][0]
              states_ex['effects'] = entity_anns[ent][1]

              ent_sent_examples[(ent, i)] = states_ex

        ex_2s['stories'][s_idx]['entities'] = [None for _ in range(len(all_entities))] # entity-story data: preconditions, effects, etc.
        for ei, ent in enumerate(all_entities):
          ent_ex = {}
          ent_ex['example_id'] = ex_2s['example_id'] + '-' + str(s_idx) + '-' + ent
          ent_ex['base_id'] = ex_2s['example_id'] + '-' + str(s_idx)
          ent_ex['sentences'] = ex['sentences']
          ent_ex['entity'] = ent
          ent_ex['attributes'] = np.zeros((max_story_length, len(att_to_idx)))
          ent_ex['preconditions'] = np.zeros((max_story_length, len(att_to_idx)))
          ent_ex['effects'] = np.zeros((max_story_length, len(att_to_idx)))
          
          ent_ex['conflict_span'] = (0,0)
          ent_ex['conflict_span_onehot'] = np.zeros((max_story_length))
          ent_ex['plausible'] = 1
          if s_idx != ex_2s['label'] and ex_2s['label'] != -1:
            # print(ex_2s['confl_sents'])
            conflict_span = (max([s+1 for s in ex_2s['confl_sents'] if s < ex_2s['breakpoint']]), ex_2s['breakpoint']+1) 
            
            # Check if the entity has some nontrivial annotated states in the boundaries of the conflict span
            if (ent, conflict_span[0]-1) in ent_sent_examples:
              for i, att in enumerate(att_default_values):
                if (ent_sent_examples[(ent, conflict_span[0]-1)]['preconditions'][i] != att_default_values[att] or ent_sent_examples[(ent, conflict_span[0]-1)]['effects'][i] != att_default_values[att]):
                  ent_ex['conflict_span'] = conflict_span
                  ent_ex['plausible'] = 0
            if (ent, conflict_span[1]-1) in ent_sent_examples:
              for i, att in enumerate(att_default_values):
                if (ent_sent_examples[(ent, conflict_span[1]-1)]['preconditions'][i] != att_default_values[att] or ent_sent_examples[(ent, conflict_span[1]-1)]['effects'][i] != att_default_values[att]):
                  ent_ex['conflict_span'] = conflict_span
                  ent_ex['plausible'] = 0

          for cs in ent_ex['conflict_span']:
            if cs > 0:
              ent_ex['conflict_span_onehot'][cs-1] = 1

          # Get binary label for each span of text as well (for alternative formulation)
          ent_ex['span_labels'] = np.zeros((max_story_length * (max_story_length - 1) // 2))
          if s_idx == 1 - ex_2s['label']: # If this is the implausible choice
            span_idx = 0
            for s2 in range(1, len(ex['sentences'])):
              for s1 in range(s2):
                # print(ex['confl_pairs'])
                for p1, p2 in ex_2s['confl_pairs']:
                  if s1 <= p1 and s2 >= p2:
                    # Check if the entity has some nontrivial annotated states in the boundaries of the conflict span
                    if (ent, p1) in ent_sent_examples:
                      for i, att in enumerate(att_default_values):
                        if ent_sent_examples[(ent, p1)]['preconditions'][i] != att_default_values[att] or ent_sent_examples[(ent, p1)]['effects'][i] != att_default_values[att]:
                          ent_ex['span_labels'][span_idx] = 1
                    if (ent, p2) in ent_sent_examples:
                      for i, att in enumerate(att_default_values):
                        if ent_sent_examples[(ent, p2)]['preconditions'][i] != att_default_values[att] or ent_sent_examples[(ent, p2)]['effects'][i] != att_default_values[att]:
                          ent_ex['span_labels'][span_idx] = 1
                span_idx += 1

          for i in range(ex_2s['length']):
            if (ent, i) in ent_sent_examples:
              ent_ex['preconditions'][i,:] = ent_sent_examples[(ent, i)]['preconditions']
              ent_ex['effects'][i,:] = ent_sent_examples[(ent, i)]['effects']
              for j, att in enumerate(att_default_values):
                if ent_ex['preconditions'][i,j] != att_default_values[att] or ent_ex['effects'][i,j] != att_default_values[att]:
                  ent_ex['attributes'][i,j] = 1
          ex_2s['stories'][s_idx]['entities'][ei] = ent_ex

  return dataset

