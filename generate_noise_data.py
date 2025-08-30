"""
python generate_noise_data.py --data_noise 0.1
"""

import argparse
import ast
import copy
import json
import os
import pdb
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import csv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Data:
    def __init__(
        self,
        meta_data_path,
        interactions_data_path,
        save_data_path,
        save_denoise_model_data_path,
        data_noise,
        history_prompt,
        rec_prompt,
        denoise_prompt,
        valid_start=0.8,
        test_start=0.9,
    ):
        self.data_noise = data_noise

        self.history_prompt = history_prompt
        self.rec_prompt = rec_prompt
        self.denoise_prompt = denoise_prompt

        self.valid_start = valid_start
        self.test_start = test_start

        self.save_data_path = save_data_path
        self.save_denoise_model_data_path = save_denoise_model_data_path
        os.makedirs(save_data_path, exist_ok=True)
        os.makedirs(save_denoise_model_data_path, exist_ok=True)

        self.metadata = self.read_json_to_list(meta_data_path)
        self.reviews = self.read_reviews_json_to_df(interactions_data_path)

        self.asin2title = self.get_asin2title()  # asin -> title

        self.reviews = self.filter_asin_wo_title(self.reviews)  # only keep asin with title
        
        print(f"Interaction Num after filtering title: {len(self.reviews)}")

        if "Movie" in os.path.basename(args.data_path):
            selected_asins = np.random.choice(self.reviews["asin"].unique(), size=20000, replace=False)
            self.reviews = self.reviews[self.reviews["asin"].isin(selected_asins)]
            self.reviews = self.process_k_core(
                self.reviews, k=11
            )
        else:
            self.reviews = self.process_k_core(self.reviews, k=5)

        self.user_to_cid, self.item_to_cid = self.generate_cid()  # user -> cid, item -> cid
        self.max_user_cid = max(self.user_to_cid.values())
        self.max_item_cid = max(self.item_to_cid.values())
        self.reviews = self.add_cid_column(self.reviews)
        print(f"User Num {self.max_user_cid}, Item num {self.max_item_cid}, Interaction Num {len(self.reviews)}")

        self.cid2title4LLM, self.cid2title4Rec = self.generate_itemcid2title()  # item cid -> title
        self.user_interacted_items_dict = self.get_interacted_items_dict()

    def get_asin2title(self):
        asin2title = {}
        for meta in tqdm(self.metadata):
            if "title" in meta.keys() and len(meta["title"]) > 1:
                if meta["title"].endswith(" VHS"):
                    meta["title"] = meta["title"][:-4]
                asin2title[meta["asin"]] = meta["title"].strip(" ")
        return asin2title

    def read_json_to_list(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(ast.literal_eval(line))

        return data

    def read_reviews_json_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        selected_columns = ["overall", "unixReviewTime", "reviewerID", "asin"]
        df = df[selected_columns]
        df = df.dropna(subset=selected_columns)


        df = df.rename(
            columns={"rating": "overall", "timestamp": "unixReviewTime", "user_id": "reviewerID", "asin": "asin"}
        )
        df["unixReviewTime"] = df["unixReviewTime"].astype(int)
        return df

    def process_k_core(self, df, k):
        while True:
            user_counts = df["reviewerID"].value_counts()
            item_counts = df["asin"].value_counts()

            less_than_k_user = user_counts[user_counts < k].index
            less_than_k_item = item_counts[item_counts < k].index

            if len(less_than_k_user) == 0 and len(less_than_k_item) == 0:
                break

            df = df[~df["reviewerID"].isin(less_than_k_user)]
            df = df[~df["asin"].isin(less_than_k_item)]

        return df
    
    def process_k_core_user(self, df, k):
        user_counts = df["reviewerID"].value_counts()
        less_than_k_user = user_counts[user_counts < k].index
        df = df[~df["reviewerID"].isin(less_than_k_user)]

        return df

    def filter_rate(self, df, rate=3):
        return df[df["overall"] >= rate]

    def filter_asin_wo_title(self, df):
        return df[df["asin"].isin(self.asin2title.keys())]

    def add_cid_column(self, df):
        df["user_id"] = df["reviewerID"].map(self.user_to_cid)
        df["item_id"] = df["asin"].map(self.item_to_cid)
        return df

    def generate_cid(self):
        users_list = self.reviews["reviewerID"].unique().tolist()
        items_list = self.reviews["asin"].unique().tolist()

        user2id = {user: idx + 1 for idx, user in enumerate(users_list)}
        item2id = {item: idx + 1 for idx, item in enumerate(items_list)}

        return user2id, item2id

    def generate_itemcid2title(self):
        cid2title4LLM = {}
        cid2title4Rec = {}

        for asin, cid in self.item_to_cid.items():
            title = self.asin2title[asin]
            cid2title4Rec[cid] = title
            if title not in cid2title4LLM.values():
                cid2title4LLM[cid] = title

        item_index_2_title_4_Rec_path = os.path.join(self.save_data_path, "id2name4Rec.json")
        item_index_2_title_path = os.path.join(self.save_data_path, "id2name.json")

        with open(item_index_2_title_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4LLM, f, ensure_ascii=False, indent=4)
        with open(item_index_2_title_4_Rec_path, "w", encoding="utf-8") as f:
            json.dump(cid2title4Rec, f, ensure_ascii=False, indent=4)

        return cid2title4LLM, cid2title4Rec

    def get_interacted_items_dict(self):
        users = dict()

        for row in self.reviews.itertuples():
            user, item = row.user_id, row.item_id
            if user not in users:
                users[user] = {"items": [], "ratings": [], "timestamps": [], "noise_items": [], "mask": []}
            users[user]["items"].append(item)
            users[user]["ratings"].append(row.overall)
            users[user]["timestamps"].append(row.unixReviewTime)

        return users

    def add_noise_to_users(self, users, item_set):
        """inplace操作，为每个用户的序列添加noise"""
        for _, user_data in users.items():
            items = user_data["items"]
            noise_items = []
            mask = []
            for item in items:
                if random.random() < self.data_noise:
                    noise_item = random.choice(item_set)
                    noise_items.append(noise_item)
                    mask.append(1)
                else:
                    noise_items.append(item)
                    mask.append(0)

            user_data["noise_items"] = noise_items
            user_data["mask"] = mask

    def recover_test_interactions(self, test_interactions):
        """对改变了test数据最后一个item的序列进行恢复"""
        for interaction in test_interactions:
            items = interaction[1]
            interaction[5][-1] = items[-1]
            interaction[2][-1] = 0

        return test_interactions

    def apply_noise_to_interactions(self, interactions):

        def replace_elements_only_one(item_list, mask_list, item_set):
            assert len(item_list) == len(mask_list)

            len_num = len(item_list)
            replace_index = random.choice(range(len_num))
            replaced_item_list = [item_list[replace_index]]
            item_list[replace_index] = random.choice(item_set)
            mask_list[replace_index] = 2

            return item_list, mask_list, replaced_item_list

        interactions = copy.deepcopy(interactions)
        all_item_ids = list(self.cid2title4Rec.keys())

        for interaction in interactions:
            item_list = interaction[5]
            mask_list = interaction[2]

            item_list, mask_list, replaced_item_list = replace_elements_only_one(item_list, mask_list, all_item_ids)

            interaction[5] = item_list
            interaction[2] = mask_list
            interaction[4] = replaced_item_list

        return interactions

    def get_noise_interactions(self):
        interactions = []
        users = self.user_interacted_items_dict

        self.add_noise_to_users(users, item_set=list(self.cid2title4Rec.keys()))

        for key in tqdm(users.keys()):
            userid = key

            items = users[key]["items"]
            timestamps = users[key]["timestamps"]
            ratings = users[key]["ratings"]
            mask = users[key]["mask"]
            noise_items = users[key]["noise_items"]
            all = list(zip(items, ratings, mask, noise_items, timestamps))

            res = sorted(all, key=lambda x: int(x[-1]))
            items, ratings, mask, noise_items, timestamps = zip(*res)
            items, ratings, mask, noise_items, timestamps = (
                list(items),
                list(ratings),
                list(mask),
                list(noise_items),
                list(timestamps),
            )

            for i in range(min(10, len(items) - 1), len(items)):
                st = max(i - 10, 0)
                interactions.append(
                    [userid, items[st : i + 1], mask[st : i + 1], [], [], noise_items[st : i + 1], int(timestamps[i])]
                )

        interactions = sorted(interactions, key=lambda x: x[-1])

        train_interactions = interactions[: int(len(interactions) * args.valid_start)]
        valid_interactions = interactions[
            int(len(interactions) * args.valid_start) : int(len(interactions) * (args.valid_start + 0.1))
        ]
        test_interactions = interactions[
            int(len(interactions) * (args.test_start)) : int(len(interactions) * (args.test_start + 0.1))
        ]

        # 对改变了test数据最后一个item的序列进行恢复
        test_interactions = self.recover_test_interactions(test_interactions)
        train_interactions_manual_noise = self.apply_noise_to_interactions(train_interactions)

        for interactions in [
            train_interactions,
            valid_interactions,
            test_interactions,
        ]:
            for interaction in interactions:
                interaction[3] = [self.cid2title4Rec[x] for x in interaction[5]]

                filtered_items = [item for item, m in zip(interaction[1], interaction[2]) if m == 1]
                replaced_item_titles = [self.cid2title4Rec[x] for x in filtered_items]
                interaction[4] = replaced_item_titles
                
        for interaction in train_interactions_manual_noise:
            interaction[3] = [self.cid2title4Rec[x] for x in interaction[5]]
            replaced_item_titles = [self.cid2title4Rec[x] for x in interaction[4]]
            interaction[4] = replaced_item_titles

        return train_interactions, train_interactions_manual_noise, valid_interactions, test_interactions

    def generate_train_LLM_data(self, df_list):

        def save_csv_json(data, filename, sample_num=-1):
            if sample_num != -1 and len(data) > sample_num:
                data = data.sample(n=sample_num, random_state=42).reset_index(drop=True)

            csv_save_path = os.path.join(
                self.save_data_path, f"{filename}{'_'+str(sample_num) if sample_num!=-1 else ''}.csv"
            )
            json_save_path = os.path.join(
                self.save_data_path, f"{filename}{'_'+str(sample_num) if sample_num!=-1 else ''}.json"
            )

            data.to_csv(csv_save_path, index=False)

            json_list = []
            for _, row in tqdm(data.iterrows()):
                L = len(row["item_titles"])

                history = self.history_prompt
                for i in range(L - 1):
                    if i == 0:
                        history += '"' + row["item_titles"][i] + '"'
                    else:
                        history += ', "' + row["item_titles"][i] + '"'

                target_movie = row["item_titles"][L - 1]
                target_movie_str = '"' + target_movie + '"'
                json_list.append(
                    {
                        "instruction": self.rec_prompt,
                        "input": f"{history}\n ",
                        "output": target_movie_str,
                    }
                )
            with open(json_save_path, "w") as f:
                json.dump(json_list, f, indent=4)

        train_df, valid_df, test_df = df_list

        save_csv_json(train_df, "train", sample_num=-1)
        save_csv_json(train_df, "train", sample_num=1000)
        save_csv_json(train_df, "train", sample_num=5000)
        save_csv_json(train_df, "train", sample_num=10000)
        save_csv_json(valid_df, "valid", sample_num=-1)
        save_csv_json(valid_df, "valid", sample_num=5000)
        save_csv_json(test_df, "test", sample_num=-1)
        save_csv_json(test_df, "test", sample_num=5000)

    def generate_train_denoise_model_data(self, train_df_manual_noise):

        def save_json(data, filename, mask, sample_num=-1):
            if sample_num != -1 and len(data) > sample_num:
                data = data.sample(n=sample_num, random_state=42).reset_index(drop=True)

            json_save_path = os.path.join(
                self.save_denoise_model_data_path, f"{filename}{'_'+str(sample_num) if sample_num!=-1 else ''}.json"
            )

            json_list = []
            for _, row in tqdm(data.iterrows()):
                noise_items_title = [title for title, m in zip(row["item_titles"], row["mask"]) if m == mask]
                like_items_title = [title for title in row["replaced_item_titles"]]
                assert len(noise_items_title) == len(like_items_title)

                history = "User Interaction Sequence: "
                for i, item in enumerate(row["item_titles"]):
                    if i == 0:
                        history += '"' + item + '"'
                    else:
                        history += ', "' + item + '"'

                if len(noise_items_title) == 0:
                    response = "None"
                else:
                    noise_item = "Noise Items: "
                    for i, item in enumerate(noise_items_title):
                        if i == 0:
                            noise_item += '"' + item + '"'
                        else:
                            noise_item += ', "' + item + '"'
                    like_item = "Suggested Replacements: "
                    for i, item in enumerate(like_items_title):
                        if i == 0:
                            like_item += '"' + item + '"'
                        else:
                            like_item += ', "' + item + '"'
                    response = noise_item + "\n" + like_item

                json_list.append(
                    {
                        "instruction": self.denoise_prompt,
                        "input": f"{history}\n ",
                        "output": response,
                    }
                )
            with open(json_save_path, "w") as f:
                json.dump(json_list, f, indent=4)

        save_json(train_df_manual_noise, "train", mask=2, sample_num=-1)

    def generate_interactions_data(self):
        train_interactions, train_interactions_manual_noise, valid_interactions, test_interactions = (
            self.get_noise_interactions()
        )

        column_names = [
            "user_id",
            "clean_item_ids",
            "mask",
            "item_titles",
            "replaced_item_titles",
            "item_ids",
            "timestamp",
        ]
        train_df = pd.DataFrame(train_interactions, columns=column_names)
        train_df_manual_noise = pd.DataFrame(train_interactions_manual_noise, columns=column_names)
        valid_df = pd.DataFrame(valid_interactions, columns=column_names)
        test_df = pd.DataFrame(test_interactions, columns=column_names)

        train_df = train_df.drop("clean_item_ids", axis=1)
        train_df_manual_noise = train_df_manual_noise.drop("clean_item_ids", axis=1)
        valid_df = valid_df.drop("clean_item_ids", axis=1)
        test_df = test_df.drop("clean_item_ids", axis=1)

        def move_item_ids_to_second_col(df):
            cols = list(df.columns)
            cols.insert(1, cols.pop(cols.index("item_ids")))
            return df[cols]

        train_df = move_item_ids_to_second_col(train_df)
        train_df_manual_noise = move_item_ids_to_second_col(train_df_manual_noise)
        valid_df = move_item_ids_to_second_col(valid_df)
        test_df = move_item_ids_to_second_col(test_df)

        df_list1 = [train_df, valid_df, test_df]
        self.generate_train_LLM_data(df_list1)
        self.generate_train_denoise_model_data(train_df_manual_noise)


if __name__ == "__main__":
    set_seed(42)

    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="/c23034/wbh/code/Noise_LLM4Rec/meta_data/amazon_game")
    parse.add_argument("--save_data_path", type=str, default="/c23034/wbh/code/Noise_LLM4Rec/data/amazon_game")

    parse.add_argument("--data_noise", type=float, default=0.1)
    parse.add_argument("--valid_start", type=float, default=0.8)
    parse.add_argument("--test_start", type=float, default=0.9)
    args = parse.parse_args()

    meta_data_path = os.path.join(args.data_path, "meta_Video_Games.json")
    interactions_data_path = os.path.join(args.data_path, "Video_Games_5.json")

    save_data_path = args.save_data_path + f"_noise_user{args.data_noise}"
    save_denoise_model_data_path = os.path.join(save_data_path, f"TDMD")

    if "Toy" == os.path.basename(args.data_path):
        history_prompt = "The user has played the following toys before: "
        rec_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
    elif "Movie" == os.path.basename(args.data_path):
        history_prompt = "The user has watched the following movies before: "
        rec_prompt = "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user."

    denoise_prompt = "You are to analyze a list of item titles provided by a user. Your task is to identify any item(s) that do not align with the main interests reflected by the majority of the items. After identifying these noise items, suggest alternative items that better match the user's interests."

    data = Data(
        meta_data_path=meta_data_path,
        interactions_data_path=interactions_data_path,
        save_data_path=save_data_path,
        save_denoise_model_data_path=save_denoise_model_data_path,
        data_noise=args.data_noise,
        history_prompt=history_prompt,
        rec_prompt=rec_prompt,
        denoise_prompt=denoise_prompt,
        valid_start=args.valid_start,
        test_start=args.test_start,
    )
    data.generate_interactions_data()
    print("Done!")
