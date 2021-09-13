import pickle
from www.utils import read_tsv, print_dict
import sys
from copy import deepcopy

att_to_idx = {'h_location': 0, 
              'conscious': 1, 
              'wearing': 2, 
              'h_wet': 3, 
              'hygiene': 4, 
              'location': 5, 
              'exist': 6, 
              'clean': 7, 
              'power': 8, 
              'functional': 9, 
              'pieces': 10, 
              'wet': 11, 
              'open': 12, 
              'temperature': 13, 
              'solid': 14, 
              'contain': 15, 
              'running': 16, 
              'moveable': 17, 
              'mixed': 18, 
              'edible': 19}
idx_to_att = {v: k for k,v in att_to_idx.items()}
human_atts = ['h_location', 'conscious', 'wearing', 'h_wet', 'hygiene']
att_to_num_classes = {
    "h_location": 3,
    "conscious": 9,
    "wearing": 9,
    "h_wet": 9,
    "hygiene": 9,
    "location": 9,
    "exist": 9,
    "clean": 9,
    "power": 9,
    "functional": 9,
    "pieces": 9,
    "wet": 9,
    "open": 9,
    "temperature": 9,
    "solid": 9,
    "contain": 9,
    "running": 9,
    "moveable": 9,
    "mixed": 9,
    "edible": 9
}

# This is valid for pre -> post, pre, and post (since 2 means true -> true and also just true)
att_default_values = {'h_location': 0, 'conscious': 2, 'wearing': 0, 'h_wet': 0, 'hygiene': 0, 'location': 0, 'exist': 2, 'clean': 0, 'power': 0, 'functional': 2, 'pieces': 0, 'wet': 0, 'open': 0, 'temperature': 0, 'solid': 0, 'contain': 0, 'running': 0, 'moveable': 2, 'mixed': 0, 'edible': 0}

# Naive way to check if a given entity name is a human
def is_human(entity):
  return (entity[0].isupper() and entity != 'TV')

att_types = {'h_location': 'h_location', 
                    'conscious': 'default', 
                    'wearing': 'default', 
                    'h_wet': 'default', 
                    'hygiene': 'default', 
                    'location': 'location', 
                    'exist': 'default', 
                    'clean': 'default',
                    'power': 'default', 
                    'functional': 'default', 
                    'pieces': 'default', 
                    'wet': 'default', 
                    'open': 'default', 
                    'temperature': 'default', 
                    'solid': 'default', 
                    'contain': 'default', 
                    'running': 'default', 
                    'moveable': 'default', 
                    'mixed': 'default', 
                    'edible': 'default'}

att_types = {'h_location': 'h_location', 
                    'conscious': 'default', 
                    'wearing': 'default', 
                    'h_wet': 'default', 
                    'hygiene': 'default', 
                    'location': 'location', 
                    'exist': 'default', 
                    'clean': 'default',
                    'power': 'default', 
                    'functional': 'default', 
                    'pieces': 'default', 
                    'wet': 'default', 
                    'open': 'default', 
                    'temperature': 'default', 
                    'solid': 'default', 
                    'contain': 'default', 
                    'running': 'default', 
                    'moveable': 'default', 
                    'mixed': 'default', 
                    'edible': 'default'}
att_change_dir = {'h_location': {0: 'does not move to a new location', 1: 'disappears', 2: 'moves somewhere new'},
            'location': {0: 'does not move to a new location', 1: 'disappears', 2: 'is picked up', 3: 'is put down', 4: 'is put on', 5: 'is removed', 6: 'is put into a container', 7: 'is taken out of a container', 8: 'moved somewhere new'},
            'default': {0: (-1,-1), 1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (0, 1), 5: (-1, 0), 6: (-1, 1), 7: (0, -1), 8: (1, -1)}}
att_change_dir_bw = {'default': {(-1, -1): 0, (0, 0): 1, (1, 1): 2, (1, 0): 3, (0, 1): 4, (-1, 0): 5, (-1, 1): 6, (0, -1): 7, (1, -1): 8}}
att_adj = { 'conscious': ('unconscious', 'conscious'),
            'wearing': ('undressed', 'dressed'), 
            'h_wet': ('dry', 'wet'), 
            'hygiene': ('dirty', 'clean'), 
            'exist': ('nonexistent', 'existent'), 
            'clean': ('dirty', 'clean'),
            'power': ('unpowered', 'powered'), 
            'functional': ('broken', 'functional'), 
            'pieces': ('whole', 'in pieces'), 
            'wet': ('dry', 'wet'), 
            'open': ('closed', 'open'), 
            'temperature': ('cold', 'hot'), 
            'solid': ('fluid', 'solid'), 
            'contain': ('empty', 'occupied'), 
            'running': ('turned off', 'turned on'), 
            'moveable': ('stuck', 'moveable'), 
            'mixed': ('separated', 'mixed'), 
            'edible': ('inedible', 'edible')}


# Converts dataset of state transition labels (too many labels) to pre-condition prediction
def extract_preconditions(dataset, att):
  if att_types[att] == 'default': # Location attributes don't have well-defined pre and post-condition - might be best to go back to span prediction for this?
    for ex in dataset:
      ex['label'] = att_change_dir['default'][ex['label']][0] + 1 # Add one for the label prediction; -1 can become 0 and so on
  return dataset

# Converts dataset of state transition labels (too many labels) to post-condition prediction
def extract_postconditions(dataset, att):
  if att_types[att] == 'default': # Location attributes don't have well-defined pre and post-condition - might be best to go back to span prediction for this?
    for ex in dataset:
      ex['label'] = att_change_dir['default'][ex['label']][1] + 1 # Add one for the label prediction; -1 can become 0 and so on
  return dataset