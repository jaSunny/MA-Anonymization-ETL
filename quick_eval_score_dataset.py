"""
    This script offers the opportunity to determine the data score iteratively
    over the increasing length of available attributes / columns for a
    given dataset.
    The scores will be logged as appended list to the scores.txt file in the
    directory output/
"""

import pandas as pd
from evaluation import *
import multiprocessing

filename = "../../extended_user_data_V2.csv.gz"
num_cores = multiprocessing.cpu_count()-1

df = pd.read_csv(filename,compression='gzip', header=0, sep=',', quotechar='"')
colnames = list(df.columns.values)

for L in range(2,82):
  df2 = df[colnames[:L]]
  df2_colnames = list(df2.columns.values)
  score = estimate_data_quality(df2,df2_colnames,num_cores)

  with open("output/scores.txt", "a") as myfile:
      myfile.write(str(round(score,2))+","+str(L)+"\n")
