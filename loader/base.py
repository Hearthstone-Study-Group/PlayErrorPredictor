#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
import pandas as pd
from loader.tags import Tags
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class BaseLoader:
    base_dir = "runtime"
    data_dir = "data"
    cached_prob_dir = "resource/data"
    name = "Base"
    score_filename = "score.txt"
    label_filename = "labels.txt"

    inner_train_data = []
    train_data = []
    test_data = []
    all_data = []

    def __init__(self, token=""):
        # Initialize directory
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        date_token = token
        if len(token) == 0:
            date_token = self.name + " "
            date_token += datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.storage_folder = os.path.join(self.base_dir, date_token)
        logging.info(f"Initialized score directory in {self.storage_folder}")
        self.ref_dir = os.path.join(self.storage_folder, "ref")
        self.res_dir = os.path.join(self.storage_folder, "res")
        if not os.path.exists(self.storage_folder):
            os.mkdir(self.storage_folder)
        else:
            logging.warning(f"Directory {self.storage_folder} already exists!")
        if not os.path.exists(self.ref_dir):
            os.mkdir(self.ref_dir)
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        # Initialize task dataset
        task = Tags(os.path.join(self.data_dir))
        task.load()
        self.all_data = task.df
        self.final_data = task.df
        self.all_categorical_data = task.df

        logging.debug(f"All data header:")
        logging.debug(self.all_data.head())
        logging.debug(f"Final evaluation data header:")
        logging.debug(self.final_data.head())

    def eval(self, labels, predictions):
        task_confusion_matrix = confusion_matrix(labels, predictions)
        task_precision = precision_score(labels, predictions)
        task_recall = recall_score(labels, predictions)
        task_f1 = f1_score(labels, predictions)

        file_path = os.path.join(self.storage_folder, self.score_filename)
        with open(file_path, "w") as score_file:
            score_file.write('task1_precision:' + str(task_precision) + '\n')
            score_file.write('task1_recall:' + str(task_recall) + '\n')
            score_file.write('task1_f1:' + str(task_f1) + '\n')

        logging.info(f"Confusion matrix")
        logging.info(task_confusion_matrix)
        logging.info(f"Precision score")
        logging.info(task_precision)
        logging.info(f"Recall score")
        logging.info(task_recall)
        logging.info(f"F1 score")
        logging.info(task_f1)
        logging.info(f"Score file written to {file_path}")

    def eval_per(self, labels, predictions, class_name, class_value):
        task_confusion_matrix = confusion_matrix(labels, predictions)
        task_precision = precision_score(labels, predictions)
        task_recall = recall_score(labels, predictions)
        task_f1 = f1_score(labels, predictions)
        filename = "score" + f'{class_name}_{class_value}.txt'
        file_path = os.path.join(self.storage_folder, filename)
        with open(file_path, "w") as score_file:
            score_file.write('task1_precision:' + str(task_precision) + '\n')
            score_file.write('task1_recall:' + str(task_recall) + '\n')
            score_file.write('task1_f1:' + str(task_f1) + '\n')
            score_file.write(f'{class_name}_{class_value} total_samples:' + str(len(labels)) + '\n')

        logging.info(f'{class_name}_{class_value} result:')
        logging.info(f"Confusion matrix")
        logging.info(task_confusion_matrix)
        logging.info(f"Precision score")
        logging.info(task_precision)
        logging.info(f"Recall score")
        logging.info(task_recall)
        logging.info(f"F1 score")
        logging.info(task_f1)
        logging.info(f"Score file written to {file_path}")

    def final(self, predictions, epoch):
        pass
