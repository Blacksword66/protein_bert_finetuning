from proteinbert import finetuning as ft
from proteinbert import load_pretrained_model, OutputType, OutputSpec
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BENCHMARK_NAME = 'signalP_binary'
BENCHMARKS_DIR = '/Users/yeyatiprasher/Coding/Internship/protein_bert/protein_benchmarks'
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

train_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.train.csv' % BENCHMARK_NAME)
train_set = pd.read_csv(train_set_file_path).dropna().drop_duplicates()
train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.1, random_state = 0)

test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % BENCHMARK_NAME)
test_set = pd.read_csv(test_set_file_path).dropna().drop_duplicates()

pretrained_model_generator, input_encoder = load_pretrained_model()



X,Y,weights = ft.encode_dataset(train_set['seq'], train_set['label'], input_encoder, OUTPUT_SPEC)

print("X = ", X)
print("Y = ",Y)
print("weights = ", weights)