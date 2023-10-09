import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def print_table(data_names, data_values):

    data = [[i, j] for i, j in zip(data_names, data_values)]

    column_widths = [max(len(str(item)) for item in column) for column in zip(*data)]

    header = ["{:{}}".format(column, width) for column, width in zip(['Parametri', 'Valore'], column_widths)]
    print(" | ".join(header))

    separator = ["-" * width for width in column_widths]
    print(" | ".join(separator))

    for row in data:
        formatted_row = ["{:{}}".format(str(item), width) for item, width in zip(row, column_widths)]
        print(" | ".join(formatted_row))


def preprocessing(df):
  
  df = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[['acc_x', 'acc_y', 'acc_z']])
  return df
  
    
def read_and_preprocessing(file_path):
  df = pd.read_parquet(file_path)
  df_processed = preprocessing(df)
  return df_processed


def parquet_generator():
  parquet_folder = r'/content/drive/MyDrive/Colab Notebooks/R&D/Scoring_Fc/TadGan/crash_file'
  files = os.listdir(parquet_folder)

  for file in files:
    file_path = os.path.join(parquet_folder, file)
    processed_data = read_and_preprocessing(file_path).reshape(-1, 750, 3)
    yield processed_data

















