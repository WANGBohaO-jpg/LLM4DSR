from collections import Counter
import copy
import pdb
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
import world
import torch.nn.functional as F
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random


def read_file(path):
    df = pd.read_csv(path)
    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    if "noise_items" in df.columns:
        df["noise_items"] = df["noise_items"].apply(ast.literal_eval)
    if "noise_items_prob" in df.columns:
        df["noise_items_prob"] = df["noise_items_prob"].apply(ast.literal_eval)
    if "suggest_item_ids" in df.columns:
        df["suggest_item_ids"] = df["suggest_item_ids"].apply(ast.literal_eval)
    if "noise_prob" in df.columns:
        df["noise_prob"] = df["noise_prob"].apply(ast.literal_eval)
    if "diverse_suggest_item_ids" in df.columns:
        df["diverse_suggest_item_ids"] = df["diverse_suggest_item_ids"].apply(ast.literal_eval)
    if "denoise_item_ids" in df.columns:
        df["denoise_item_ids"] = df["denoise_item_ids"].apply(ast.literal_eval)
        df["item_ids"] = df["denoise_item_ids"]

    return df


def trans_into_denoise_df(df, threshold, max_denoise_num):
    def process_row(row):
        item_ids = copy.deepcopy(row["item_ids"])
        noise_items = row["noise_items"][:max_denoise_num]
        noise_items_prob = row["noise_items_prob"]
        suggest_item_ids = row["suggest_item_ids"][:max_denoise_num]

        id_counts = Counter(item_ids)

        for i, id in enumerate(item_ids):
            if id in noise_items and noise_items_prob[noise_items.index(id)] > threshold:
                item_ids[i] = suggest_item_ids[noise_items.index(id)]
            row["item_ids"] = item_ids

        return row

    df = df.apply(process_row, axis=1)
    return df


class Data_Pro:
    def __init__(self, dataroot, train_denoise_flag=False, test_denoise_flag=False):
        train_file = os.path.join(dataroot, "train.csv")
        test_file = os.path.join(dataroot, "test_5000.csv")

        train_df, test_df = read_file(train_file), read_file(test_file)
        train_size, test_size = len(train_df), len(test_df)
        data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        if "noise_prob" in data_df.columns:
            all_scores = [score for sublist in train_df["noise_prob"] for score in sublist]
            train_noise_scores_threshold = np.percentile(all_scores, world.config["train_noise_filter_threshold"] * 100)

            all_scores = [score for sublist in test_df["noise_prob"] for score in sublist]
            test_noise_scores_threshold = np.percentile(all_scores, world.config["test_noise_filter_threshold"] * 100)
        else:
            train_noise_scores_threshold = 0
            test_noise_scores_threshold = 0

        print(f"train_noise_threshold: {train_noise_scores_threshold}")
        print(f"test_noise_threshold: {test_noise_scores_threshold}")

        self.train_df, self.test_df = (data_df.iloc[:train_size], data_df.iloc[-test_size:])

        max_item_id = max(data_df["item_ids"].apply(max))
        self.item_num = max_item_id
        print("Max Item ID: ", self.item_num)
        print("Max User ID: ", max(data_df["user_id"]))

        self.train_df = (
            trans_into_denoise_df(
                self.train_df,
                threshold=train_noise_scores_threshold,
                max_denoise_num=world.config["train_max_denoise_num"],
            )
            if train_denoise_flag
            else self.train_df
        )
        self.test_df = (
            trans_into_denoise_df(
                self.test_df,
                threshold=test_noise_scores_threshold,
                max_denoise_num=world.config["test_max_denoise_num"],
            )
            if test_denoise_flag
            else self.test_df
        )

    def get_data_df(self):
        return self.train_df, self.test_df


class Seq_dataset(Dataset):
    def __init__(self, data_frame):
        self.df = data_frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_ids = row["item_ids"]

        seq_item_id = torch.tensor(item_ids, dtype=torch.long) if type(item_ids) is list else item_ids
        seq_item_id = F.pad(seq_item_id, (world.config["maxlen"] + 1 - seq_item_id.size(0), 0), "constant", 0)

        return seq_item_id
