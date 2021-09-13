import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys
import pickle
import os

def print_dict(ex):
  print('{')
  for k, v in ex.items():
    print('  ' + str(k) + ': ')
    if type(v) == type([]) and len(v) > 0:
      print('    [')
      for elem in v:
        print('      ' + str(elem))
      print('    ],')
    else:
      print('    ' + str(v) + ',')
  print('}')

  print('\n')

# Borrowed from https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=gpt6tR83keZD
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_model_dir(name, subtask, bs, lr, e):
  return '%s_%s_%s_%s_%s' % (str(name), str(subtask), str(bs), str(lr), str(e))

def read_tsv(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

# Returns a list of indices for first instance where sl appears within l (empty is sl is not a sublist of l)
def get_sublist(l, sl):
  len_l = len(l)
  len_sl = len(sl)
  for i, t in enumerate(l[:len_l-len_sl]):
    idx = []
    if t == sl[0]:
      idx.append(i)
      for j in range(1, len_sl):
        if l[i+j] != sl[j]:
          break # Sublist isn't here
        idx.append(i+j)
      return idx
  return [] # If we made it here, sublist is not within list
      