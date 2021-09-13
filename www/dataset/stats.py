from collections import Counter
import numpy as np
import spacy

def get_dataset_stats(dataset):

  nlp = spacy.load("en_core_web_sm")

  # Plug in which variables from JSON data we want to track
  count_vars = {'aug': True}
  dist_vars = ['type', 
               'location', 
               'breakpoint',
               ]
  avg_vars = ['length']
  stories_per_vars = ['worker_id'] # e.g., stories per worker
  vocab_vars = ['sentences']

  stats = {p: {} for p in dataset}
  for p in dataset:
    stats[p]['count'] = {str(k) + '=' + str(v): 0 for k,v in count_vars.items()}
    stats[p]['dist'] = {k: Counter() for k in dist_vars}
    stats[p]['avg'] = {k: 0.0 for k in avg_vars}
    stats[p]['stories_per'] = {k: Counter() for k in stories_per_vars}
    stats[p]['vocab'] = {k: {'tokens': Counter(), 'lemmas': Counter(), 'tags': Counter(), 'nouns': Counter(), 'verbs': Counter()} for k in vocab_vars}
    
    # i = 0
    for ex in dataset[p]:
      # i += 1
      # if i > 10:
      #   break

      exid = ex['example_id']

      for k, v in count_vars.items():
        if ex[k] == v:
          stats[p]['count'][str(k) + '=' + str(v)] += 1

      for k in dist_vars:
        v = ex[k]
        if v in stats[p]['dist']:
          stats[p]['dist'][k][v] = 1
        else:
          stats[p]['dist'][k][v] += 1

      for k in avg_vars:
        stats[p]['avg'][k] += ex[k]

      for k in stories_per_vars:
        v = ex[k]
        if v in stats[p]['stories_per']:
          stats[p]['stories_per'][k][v] = 1
        else:
          stats[p]['stories_per'][k][v] += 1

      # Get vocab (words + verbs + nouns) from data
      for k in vocab_vars:
        for s in ex[k]:
          doc = nlp(s)
          stats[p]['vocab'][k]['tokens'] += Counter([t.text for t in doc])
          stats[p]['vocab'][k]['lemmas'] += Counter([t.lemma_ for t in doc])
          stats[p]['vocab'][k]['tags'] += Counter([t.pos_ for t in doc])
          stats[p]['vocab'][k]['nouns'] += Counter([t.lemma for t in doc if t.pos_.startswith('N')])
          stats[p]['vocab'][k]['verbs'] += Counter([t.lemma for t in doc if t.pos_.startswith('V')])

    # Finish avg calculations
    for k in avg_vars:
      stats[p]['avg'][k] /= float(len(dataset[p]))

  # Aggregate stats into overall values
  stats['overall'] = {}

  stats['overall']['count'] = {}
  for k, v in count_vars.items():
    total = 0
    for p in dataset:
      total += stats[p]['count'][str(k) + '=' + str(v)]
    stats['overall']['count'][str(k) + '=' + str(v)] = total

  stats['overall']['dist'] = {}
  for k in dist_vars:
    total = Counter()
    for p in dataset:
      total += stats[p]['dist'][k]
    stats['overall']['dist'][k] = total

  stats['overall']['avg'] = {}
  for k in avg_vars:
    total = 0.0
    total_count = 0
    for p in dataset:
      total += len(dataset[p]) * stats[p]['avg'][k]
      total_count += len(dataset[p])
    total /= float(total_count)
    stats['overall']['avg'][k] = total

  stats['overall']['stories_per'] = {}
  for k in stories_per_vars:
    total = Counter()
    for p in dataset:
      total += stats[p]['stories_per'][k]
    stats['overall']['stories_per'][k] = total

  stats['overall']['vocab'] = {}
  for p in dataset:
    for k in vocab_vars:
      if k not in stats['overall']['vocab']:
        stats['overall']['vocab'][k] = {}
      for t in stats[p]['vocab'][k]:
        if t not in stats['overall']['vocab'][k]:
          stats['overall']['vocab'][k][t] = Counter()
        stats['overall']['vocab'][k][t] += stats[p]['vocab'][k][t]

  # Finish stories per calculations
  for p in stats:
    for k in stories_per_vars:
      stats[p]['stories_per'][k] = np.mean([v for k, v in stats[p]['stories_per'][k].items()])

  return stats