#!/usr/bin/env python
import sys
import os
import os.path

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
import pandas as pd
from model.BackTranslate import BackTranslate
from loader.custom import CustomLoader
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from collections import OrderedDict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import tqdm

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class OfficialLoader(CustomLoader):
    name = "Split"

    def __init__(self, token=""):
        super().__init__(token)

    def split(self):
        self.train_data = []  # par_id, label and text
        for idx in range(len(self.all_data)):
            id = self.all_data.id[idx]
            description = self.all_data.text[idx]
            requirements = self.all_data.requirements[idx]
            self.train_data.append({
                'id': id,
                'text': description,
                'label': requirements
            })
        self.train_data = pd.DataFrame(self.train_data)
        self.test_data = []  # par_id, label and text
        for idx in range(len(self.all_data)):
            id = self.all_data.id[idx]
            description = self.all_data.text[idx]
            requirements = self.all_data.requirements[idx]
            self.test_data.append({
                'id': id,
                'text': description,
                'label': requirements
            })
        self.test_data = pd.DataFrame(self.test_data)
        # self.train_data.to_csv(os.path.join(self.data_dir, self.official_train_data_filename))
        # self.test_data.to_csv(os.path.join(self.data_dir, self.official_test_data_filename))
        logging.info(f"Successfully split TEST({len(self.train_data)})/DEV({len(self.test_data)}).")

    def balance(self):
        pass
        # positive_class = self.train_data[self.train_data.label == 1]
        # negative_class = self.train_data[self.train_data.label == 0]
        # positive_label = len(positive_class)
        # negative_label = len(negative_class)
        # minimum_label = min(positive_label, negative_label)
        # positive_class = positive_class[:minimum_label]
        # negative_class = negative_class[:minimum_label]
        # self.train_data = pd.concat([positive_class, negative_class])
        # self.train_data = self.train_data.sample(frac=1, axis=1).reset_index(drop=True)

    def augmentation(self):
        pass
        # if os.path.isfile(os.path.join(self.data_dir, self.augmentation_data_filename)):
        #     logging.info(
        #         f"Using cached balanced dataset from {os.path.join(self.data_dir, self.augmentation_data_filename)}")
        #     self.train_data = pd.read_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
        #     logging.info(
        #         f"Cached dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
        #     return
        # positive_class = self.train_data[self.train_data.label == 1]
        # negative_class = self.train_data[self.train_data.label == 0]
        # positive_label = len(positive_class)
        # negative_label = len(negative_class)
        # minimum_label = min(positive_label, negative_label)
        # maximum_label = max(positive_label, negative_label)
        # target_class = positive_class
        # other_class = negative_class
        # target_label = 1
        # if minimum_label == negative_label:
        #     target_class = negative_class
        #     other_class = positive_class
        #     target_label = 0
        # sources = target_class["text"].to_list()
        # print(sources[:5])
        # target_langs = ["fr", "es", "it", "pt", "ro", "ca", "gl", "la"]
        # model = BackTranslate()
        # for round in range(int(maximum_label / minimum_label) + 1):
        #     target_lang = target_langs[round % len(target_langs)]
        #     logging.info(f"Data augumentation round {round}, intermediate language {target_lang}")
        #     translated = model.back_translate(sources, target_lang=target_lang)
        #     insertion = []
        #     for text in translated:
        #         insertion.append({
        #             'par_id': 0,
        #             'text': text,
        #             'label': target_label
        #         })
        #     logging.info(f"Generated {len(insertion)} ({target_label}) samples")
        #     target_class = target_class.append(insertion)
        # logging.info(f"After augumentation: {len(target_class)}/{len(other_class)} samples")
        # self.train_data = pd.concat([target_class, other_class])
        # self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
        # self.train_data = self.train_data.sample(frac=1, axis=1).reset_index(drop=True)

    def process(self, input_name, output_name):
        pass
        # if os.path.isfile(os.path.join(self.data_dir, output_name)):
        #     logging.info(
        #         f"[process] output file exists: {os.path.join(self.data_dir, output_name)}")
        #     return
        # if os.path.isfile(os.path.join(self.data_dir, input_name)):
        #
        #     logging.info(
        #         f"[process] input from {os.path.join(self.data_dir, input_name)}")
        #     if input_name[-3:] == 'tsv':
        #         self.train_data = pd.read_csv(os.path.join(self.data_dir, input_name), sep='\t',
        #                                       names=['par_id', 'art_id', 'keyword', 'country', 'text'])
        #     else:
        #         self.train_data = pd.read_csv(os.path.join(self.data_dir, input_name))
        #     print((self.train_data.head()))
        #     count = 0
        #     for idx, row in self.train_data.iterrows():
        #         if not isinstance(row['text'], str):
        #             continue
        #         if len(row['text']) > 512:
        #             count += 1
        #             self.train_data.loc[idx, 'text'] = row['text'][:128] + row['text'][-384:]
        #             # row['text'] = row['text'][:128] + row['text'][-384:]
        #     print(f'[process] before truncation, count = {count}\n')
        #     count = 0
        #     for idx, row in self.train_data.iterrows():
        #         if not isinstance(row['text'], str):
        #             continue
        #         if len(row['text']) > 512:
        #             count += 1
        #     print(f'[process] after truncation, count = {count}\n')
        #     print((self.train_data.head()))
        #     if 'Unnamed: 0' in self.train_data.columns:
        #         self.train_data = self.train_data.drop(columns=['Unnamed: 0'])
        #     print((self.train_data.head()))
        #     self.train_data.to_csv(os.path.join(self.data_dir, output_name))
