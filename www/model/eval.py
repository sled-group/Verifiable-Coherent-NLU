import numpy as np
import time
from www.utils import format_time, read_tsv
import torch
import json
import os
import progressbar
from www.dataset.ann import att_default_values

# Run evaluation for a PyTorch model
def evaluate(model, eval_dataloader, device, metrics, list_output=False, num_outputs=1, span_mode=False, seg_mode=False, return_softmax=False, multilabel=False, lm=False):
  print('\tBeginning evaluation...')

  t0 = time.time()

  model.zero_grad()
  model.eval()

  all_labels = None
  if not list_output:
    all_preds = None
    if return_softmax:
      all_logits = None
  else:
    all_preds = [np.array([]) for _ in range(num_outputs)]

  print('\t\tRunning prediction...')
  # Get preds from model
  for batch in eval_dataloader:
    
    # Move to GPU
    batch = tuple(t.to(device) for t in batch)

    if span_mode:
      input_ids, input_mask, labels, spans = batch
    elif seg_mode:
      input_ids, input_mask, labels, segment_ids = batch
    else:
      input_ids, input_mask, labels = batch

    with torch.no_grad():
      if span_mode:
        out = model(input_ids,
                    token_type_ids=None,
                    attention_mask=input_mask,
                    spans=spans)
      elif seg_mode:
        out = model(input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask)
      else:        
        out = model(input_ids,
                    token_type_ids=None,
                    attention_mask=input_mask)
    
    label_ids = labels.to('cpu').numpy()
    if all_labels is None:
      all_labels = label_ids
    else:
      all_labels = np.concatenate((all_labels, label_ids), axis=0)
    # print(all_labels.shape)

    logits = out[0]
    if list_output: # This is broken, do not use
      metr = {}
      for o in range(num_outputs):
        preds = torch.argmax(logits[o], dim=1).detach().cpu().numpy()
        all_preds[o] = np.concatenate((all_preds[o], preds))
        metr_o = compute_metrics(all_preds[o], all_labels, metrics)
        for k, v in metr_o.items():
          metr[str(o) + '_' + k] = v    
    elif multilabel:
      preds = torch.sigmoid(logits)
      preds[preds > 0.5] = 1
      preds[preds < 0.5] = 0
      preds = preds.detach().cpu().numpy()
      if all_preds is None:
        all_preds = preds
      else:
        all_preds = np.concatenate((all_preds, preds), axis=0)
      
      if return_softmax:
        if not multilabel:
          logits = torch.softmax(logits, dim=1).detach().cpu().numpy() 
        else:
          logits = torch.sigmoid(logits).detach().cpu().numpy() 
        if all_logits is None:
          all_logits = logits
        else:
          all_logits = np.concatenate((all_logits, logits))      
      
    else:
      preds = torch.argmax(logits, dim=1 if not lm else 2).detach().cpu().numpy()
      # print(preds.shape)
      if all_preds is None:
        all_preds = preds
      else:
        all_preds = np.concatenate((all_preds, preds))
      # print(all_preds.shape)
      if return_softmax:
        logits = torch.softmax(logits, dim=1 if not lm else 2).detach().cpu().numpy() 
        if all_logits is None:
          all_logits = logits
        else:
          all_logits = np.concatenate((all_logits, logits))

  # Calculate metrics
  print('\t\tComputing metrics...')
  if multilabel:
    # Overall metrics and per-label metrics
    metr = compute_metrics(all_preds.flatten(), all_labels.flatten(), metrics)
    for i in range(model.num_labels):
      metr_i = compute_metrics(all_preds.reshape(-1, model.num_labels)[:, i], all_labels.reshape(-1, model.num_labels)[:, i].flatten(), metrics)
      for k in metr_i:
        metr['%s_%s' % (str(k), str(i))] = metr_i[k]
  elif lm:
    # In language modeling, ignore examples where label is -100 and flatten
    print('\t\t\tFlattening and filtering preds and labels for LM...')
    preds_temp = all_preds.flatten()
    labels_temp = all_labels.flatten()
    metr = compute_metrics(preds_temp[labels_temp != -100], labels_temp[labels_temp != -100], metrics)
  else:
    metr = compute_metrics(all_preds, all_labels, metrics)

  print('\tFinished evaluation in %ss.' % str(format_time(time.time() - t0)))

  if not return_softmax:
    return metr, all_preds, all_labels
  else:
    # Warning: this is not supported in list_output mode
    return metr, all_preds, all_labels, all_logits


def compute_metrics(preds, labels, metrics):
  metr = {}
  for m, m_name in metrics:
    if m_name in ['accuracy', 'confusion_matrix']:
      metr[m_name] = m(labels, preds) # Assume each metric m will be a function of (y_true, y_pred)
    else:
      metr[m_name] = m(labels, preds, average='macro')
  return metr

# Save eval metrics (or any dictionary) to json file
def save_results(results, output_dir, dataset_name):
  with open(os.path.join(output_dir, 'results_%s.json' % str(dataset_name)), 'w') as f:
    json.dump(results, f)

# Print eval preds for a model on some dataset
def save_preds(ids, labels, preds, output_dir, dataset_name):
  assert len(ids) == len(labels) == len(preds)
  if len(labels.shape) == 1:
    with open(os.path.join(output_dir, 'preds_%s.tsv' % str(dataset_name)), 'w') as f:
      for exid, label, pred in zip(ids, labels, preds):
        f.write(exid + '\t' + str(int(label)) + '\t' + str(int(pred)) + '\n')
  else:
    with open(os.path.join(output_dir, 'preds_%s.tsv' % str(dataset_name)), 'w') as f:
      for exid, label, pred in zip(ids, labels, preds):
        f.write(exid + '\t' + '\t'.join([str(int(l)) for l in label]) + '\t' + '\t'.join([str(int(p)) for p in pred]) + '\n')

# Print eval probs (softmax) for a model on some dataset
def save_probs(ids, labels, preds, output_dir, dataset_name):
  assert len(ids) == len(labels) == len(preds)
  if len(labels.shape) == 1:
    with open(os.path.join(output_dir, 'probs_%s.tsv' % str(dataset_name)), 'w') as f:
      for exid, label, pred in zip(ids, labels, preds):
        f.write(exid + '\t' + str(int(label)) + '\t' + '\t'.join([str(p) for p in pred]) + '\n')
  else:
    with open(os.path.join(output_dir, 'probs_%s.tsv' % str(dataset_name)), 'w') as f:
      for exid, label, pred in zip(ids, labels, preds):
        f.write(exid + '\t' + '\t'.join([str(int(l)) for l in label]) + '\t' + '\t'.join([str(p) for p in pred]) + '\n')

# Load model predictions (id, pred, label) from file
def load_preds(fname):
  lines = read_tsv(fname)
  preds = {}
  for l in lines:
    exid, label, pred = l[0], int(float(l[1])), int(float(l[2]))
    preds[exid] = {'label': label, 'pred': pred}
  return preds

# Get some generic metrics for a predicted and g.t. list of integers
def list_comparison(pred, label):

  if len(pred) > 0:
    prec = len([p for p in pred if p in label]) / len(pred) 
  else:
    prec = None
  
  if len(label) > 0:
    rec = len([p for p in pred if p in label]) / len(label)
  else:
    rec = None

  # Define a "correct" prediction as a prediction with no incorrect values, 
  # which has at least one value if the label is non-empty
  #
  # Define a "perfect" prediction as an exact match of label and pred
  if len(label) == 0:
    if len(pred) == 0:
      corr = True
      perf = True
    else:
      corr = False
      perf = False
  else:
    if len(pred) > 0 and len([p for p in pred if p not in label]) == 0:
      corr = True
    else:
      corr = False

    if len(pred) > 0 and set(pred) == set(label):
      perf = True
    else:
      perf = False

  return prec, rec, corr, perf

# Run evaluation for the conflict detector
def evaluate_tiered(model, eval_dataloader, device, metrics, seg_mode=False, return_softmax=False, return_explanations=False, return_losses=False, verbose=True):
  if verbose:
    print('\tBeginning evaluation...')

  t0 = time.time()

  model.zero_grad()
  model.eval()
  for layer in model.precondition_classifiers:
    layer.eval()
  for layer in model.effect_classifiers:
    layer.eval()    

  all_pred_attributes = None
  all_attributes = None

  all_pred_prec = None
  all_prec = None

  all_pred_eff = None
  all_eff = None

  all_pred_conflicts = None
  all_conflicts = None

  all_pred_stories = None
  all_stories = None  
  if return_softmax:
    all_prob_stories
  
  if verbose:
    print('\t\tRunning prediction...')

  if verbose:
    bar_size = len(eval_dataloader)
    bar = progressbar.ProgressBar(max_value=bar_size, widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar_idx = 0
    bar.start()

  # Aggregate losses
  agg_losses = {}

  # Get preds from model
  for batch in eval_dataloader:
    # Move to GPU
    batch = tuple(t.to(device) for t in batch)

    input_ids = batch[0].long().to(device)
    input_lengths = batch[1].to(device)
    input_entities = batch[2].to(device)
    input_mask = batch[3].to(device)
    attributes = batch[4].long().to(device)
    preconditions = batch[5].long().to(device)
    effects = batch[6].long().to(device)
    conflicts = batch[7].long().to(device)
    labels = batch[8].long().to(device)
    if seg_mode:
      segment_ids = batch[9].to(device)
    else:
      segment_ids = None

    batch_size, num_stories, num_entities, num_sents, seq_length = input_ids.shape

    with torch.no_grad():
      # out = model(input_ids,
      #             input_lengths,
      #             input_entities,
      #             attention_mask=input_mask,
      #             token_type_ids=segment_ids)
      out = model(input_ids,
                  input_lengths,
                  input_entities,
                  attention_mask=input_mask,
                  token_type_ids=segment_ids,
                  attributes=attributes,
                  preconditions=preconditions,
                  effects=effects,
                  conflicts=conflicts,
                  labels=labels)
    if return_losses:
      for k in out:
        if 'loss' in k:
          if k not in agg_losses:
            agg_losses[k] = out[k]
          else:
            agg_losses[k] += out[k]

    # Get gt/predicted attributes
    if 'attributes' not in model.ablation:
      label_ids = attributes.view(-1, attributes.shape[-1]).to('cpu').numpy()
      if all_attributes is None:
        all_attributes = label_ids
      else:
        all_attributes = np.concatenate((all_attributes, label_ids), axis=0)

      preds = out['out_attributes'].detach().cpu().numpy()
      preds[preds >= 0.5] = 1
      preds[preds < 0.5] = 0
      if all_pred_attributes is None:
        all_pred_attributes = preds
      else:
        all_pred_attributes = np.concatenate((all_pred_attributes, preds), axis=0)


    # Get gt/predicted preconditions
    label_ids = preconditions.view(-1, preconditions.shape[-1]).to('cpu').numpy()
    if all_prec is None:
      all_prec = label_ids
    else:
      all_prec = np.concatenate((all_prec, label_ids), axis=0)

    preds = out['out_preconditions'].detach().cpu().numpy()
    if all_pred_prec is None:
      all_pred_prec = preds
    else:
      all_pred_prec = np.concatenate((all_pred_prec, preds), axis=0)


    # Get gt/predicted preconditions
    label_ids = effects.view(-1, effects.shape[-1]).to('cpu').numpy()
    if all_eff is None:
      all_eff = label_ids
    else:
      all_eff = np.concatenate((all_eff, label_ids), axis=0)

    preds = out['out_effects'].detach().cpu().numpy()
    if all_pred_eff is None:
      all_pred_eff = preds
    else:
      all_pred_eff = np.concatenate((all_pred_eff, preds), axis=0)


    # Get gt/predicted conflict points
    label_ids = conflicts.to('cpu').numpy()
    if all_conflicts is None:
      all_conflicts = label_ids
    else:
      all_conflicts = np.concatenate((all_conflicts, label_ids), axis=0)

    # preds_start = torch.argmax(out['out_start'],dim=-1).detach().cpu().numpy()
    # preds_end = torch.argmax(out['out_end'],dim=-1).detach().cpu().numpy()
    # preds = np.stack((preds_start, preds_end), axis=1)

    preds = out['out_conflicts'].detach().cpu().numpy()
    preds[preds < 0.5] = 0.
    preds[preds >= 0.5] = 1.
    if all_pred_conflicts is None:
      all_pred_conflicts = preds
    else:
      all_pred_conflicts = np.concatenate((all_pred_conflicts, preds), axis=0)


    # Get gt/predicted story choices
    label_ids = labels.to('cpu').numpy()
    if all_stories is None:
      all_stories = label_ids
    else:
      all_stories = np.concatenate((all_stories, label_ids), axis=0)

    preds = torch.argmax(out['out_stories'], dim=-1).detach().cpu().numpy()
    if all_pred_stories is None:
      all_pred_stories = preds
    else:
      all_pred_stories = np.concatenate((all_pred_stories, preds), axis=0)
    if return_softmax:
      probs = torch.softmax(out['out_stories'], dim=-1).detach().cpu().numpy()
      if all_prob_stories is None:
        all_prob_stories = probs
      else:
        all_prob_stories = np.concatenate((all_prob_stories, probs), axis=0)

    if verbose:
      bar_idx += 1
      bar.update(bar_idx)
  if verbose:
    bar.finish()

  # Calculate metrics
  if verbose:
    print('\t\tComputing metrics...')

  # print(all_pred_attributes.shape)
  # print(all_attributes.shape)
  # print(all_pred_prec.shape)
  # print(all_prec.shape)
  # print(all_pred_eff.shape)
  # print(all_eff.shape)
  # print(all_pred_conflicts.shape)
  # print(all_conflicts.shape)
  # print(all_pred_stories.shape)
  # print(all_stories.shape)

  input_lengths = input_lengths.detach().cpu().numpy()

  # Overall metrics and per-category metrics for attributes, preconditions, and effects
  # NOTE: there are a lot of extra negative examples due to padding along sentene and entity dimenions. This can't affect F1, but will affect accuracy and make it disproportionately large.
  metr_attr = None
  if 'attributes' not in model.ablation:
    metr_attr = compute_metrics(all_pred_attributes.flatten(), all_attributes.flatten(), metrics)
    for i in range(model.num_attributes):
      metr_i = compute_metrics(all_pred_attributes[:, i], all_attributes[:, i], metrics)
      for k in metr_i:
        metr_attr['%s_%s' % (str(k), str(i))] = metr_i[k]

  metr_prec = compute_metrics(all_pred_prec.flatten(), all_prec.flatten(), metrics)
  for i in range(model.num_attributes):
    metr_i = compute_metrics(all_pred_prec[:, i], all_prec[:, i], metrics)
    for k in metr_i:
      metr_prec['%s_%s' % (str(k), str(i))] = metr_i[k]

  metr_eff = compute_metrics(all_pred_eff.flatten(), all_eff.flatten(), metrics)
  for i in range(model.num_attributes):
    metr_i = compute_metrics(all_pred_eff[:, i], all_eff[:, i], metrics)
    for k in metr_i:
      metr_eff['%s_%s' % (str(k), str(i))] = metr_i[k]

  # Conflict span metrics
  metr_conflicts = compute_metrics(all_pred_conflicts.flatten(), all_conflicts.flatten(), metrics)

  # metr_start = compute_metrics(all_pred_spans[:,0], all_spans[:,0], metrics)
  # for k in metr_start:
  #   metr[k + '_start'] = metr_start[k]

  # metr_end = compute_metrics(all_pred_spans[:,1], all_spans[:,1], metrics)
  # for k in metr_end:
  #   metr[k + '_end'] = metr_end[k]

  metr_stories = compute_metrics(all_pred_stories.flatten(), all_stories.flatten(), metrics)

  verifiability, explanations = verifiable_reasoning(all_stories, all_pred_stories, all_conflicts, all_pred_conflicts, all_prec, all_pred_prec, all_eff, all_pred_eff, return_explanations=True)
  metr_stories['verifiability'] = verifiability

  if verbose:
    print('\tFinished evaluation in %ss.' % str(format_time(time.time() - t0)))

  return_base = [metr_attr, all_pred_attributes, all_attributes, metr_prec, all_pred_prec, all_prec, metr_eff, all_pred_eff, all_eff, metr_conflicts, all_pred_conflicts, all_conflicts, metr_stories, all_pred_stories, all_stories]
  if return_softmax:
    return_base += [all_prob_stories]
  if return_explanations:
    return_base += [explanations]
  if return_losses:
    for k in agg_losses:
      if 'loss' in k:
        agg_losses[k] /= len(eval_dataloader)
    return_base += [agg_losses]
  
  return tuple(return_base)


# "Verifiability" metric: % of examples where
# 1) Story prediction is correct
# 2) Conflicting sentences are correct
# 3) All nontrivial predicted states in the conflicting sentences are correct
def verifiable_reasoning(stories, pred_stories, conflicts, pred_conflicts, preconditions, pred_preconditions, effects, pred_effects, return_explanations=False):
  atts = list(att_default_values.keys())

  verifiable = 0
  total = 0
  explanations = []
  for i, ex in enumerate(stories):
    l_story = stories[i]
    p_story = pred_stories[i]

    l_conflict = np.sum(conflicts, axis=(1,2))[i]
    p_conflict = np.sum(pred_conflicts.reshape(conflicts.shape), axis=(1,2))[i]
    l_conflict = np.nonzero(l_conflict)[0]
    p_conflict = np.nonzero(p_conflict)[0]

    l_prec = preconditions.reshape(list(conflicts.shape[:4]) + [preconditions.shape[-1]])[i,1-l_story] # (num entities, num sentences, num attributes)
    p_prec = pred_preconditions.reshape(list(conflicts.shape[:4]) + [preconditions.shape[-1]])[i,1-l_story] # (num entities, num sentences, num attributes)

    l_eff = effects.reshape(list(conflicts.shape[:4]) + [effects.shape[-1]])[i,1-l_story] # (num entities, num sentences, num attributes)
    p_eff = pred_effects.reshape(list(conflicts.shape[:4]) + [effects.shape[-1]])[i,1-l_story] # (num entities, num sentences, num attributes)

    explanation = {'story_label': int(l_story),
                   'story_pred': int(p_story),
                   'conflict_label': [int(c) for c in l_conflict],
                   'conflict_pred': [int(c) for c in p_conflict],
                   'preconditions_label': l_prec,
                   'preconditions_pred': p_prec,
                   'effects_label': l_eff,
                   'effects_pred': p_eff,
                   'valid_explanation': False}

    if l_story == p_story:
      if len(l_conflict) == len(p_conflict) == 2:
        if l_conflict[0] == p_conflict[0] and l_conflict[1] == p_conflict[1]:
          states_verifiable = True
          found_states = False

          # Check that effect of first conflict sentence has states which are correct
          for sl, sp in [(l_eff, p_eff)]: # Check preconditions and effects
            for sl_e, sp_e in zip(sl, sp): # Check all entities
              for si in [l_conflict[0]]: # Check conflicting sentences
                sl_es = sl_e[si]
                sp_es = sp_e[si]
                for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                  if p != att_default_values[atts[j]] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                    found_states = True
                    if p != sl_es[j]:
                      states_verifiable = False

          # Check that precondition of second conflict sentence has states which are correct
          for sl, sp in [(l_prec, p_prec)]: # Check preconditions and effects
            for sl_e, sp_e in zip(sl, sp): # Check all entities        
              for si in [l_conflict[1]]: # Check conflicting sentences
                sl_es = sl_e[si]
                sp_es = sp_e[si]
                for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                  if p != att_default_values[atts[j]] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                    found_states = True
                    if p != sl_es[j]:
                      states_verifiable = False

          if states_verifiable and found_states:
            verifiable += 1
            explanation['valid_explanation'] = True

    total += 1
    explanations.append(explanation)

  if not return_explanations:
    return verifiable / total
  else:
    return verifiable / total, explanations
  
# Adds entity and attribute labels to explanations object returned from eval function (for easier to read physical states)
def add_entity_attribute_labels(explanations, dataset, attributes):
  for x, expl in enumerate(explanations):
    ex = dataset[x]
    bad_story = ex['stories'][1-ex['label']]
    expl['example_id'] = ex['example_id']
    expl['story0'] = '\n'.join(ex['stories'][0]['sentences'])
    expl['story1'] = '\n'.join(ex['stories'][1]['sentences'])
    assert ex['label'] == expl['story_label'], "mismatch between explanations and original examples!"
    
    entities = [d['entity'] for d in bad_story['entities']]
    for key in ['preconditions_label', 'preconditions_pred', 'effects_label', 'effects_pred']:
      new_states = {}
      for i, ent_anns in enumerate(expl[key]):
        if i < len(entities):
          ent = entities[i]
          new_states[ent] = {}
          for j, sent_anns in enumerate(ent_anns):
            if j < len(bad_story['sentences']):
              new_states[ent][j] = {}
              for k, att_ann in enumerate(sent_anns):
                if int(att_ann) != att_default_values[attributes[k]] and int(att_ann) > 0:
                  att = attributes[k]
                  new_states[ent][j][att] = int(att_ann)
      expl[key] = new_states
    explanations[x] = expl
  return explanations
