"""
python main.py --dataset Movie_denoise_user0.0_5000_2000
"""

import json
import os
import random

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"

import glob
import pdb
import time
import pandas as pd
import torch

from tqdm import tqdm
import copy
import nni

from model import SASRec
import world
from logger import CompleteLogger
import utils
from os.path import join
from tensorboardX import SummaryWriter
from pprint import pprint

from dataloader import Seq_dataset, Data_Pro
from torch.utils.data import DataLoader

if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================


denoise_category = world.config["denoise_category"]
if denoise_category == "all":
    train_denoise_flag = 1
    test_denoise_flag = 1
elif denoise_category == "train":
    train_denoise_flag = 1
    test_denoise_flag = 0
elif denoise_category == "test":
    train_denoise_flag = 0
    test_denoise_flag = 1
elif denoise_category == "none":
    train_denoise_flag = 0
    test_denoise_flag = 0

dataroot = os.path.join("./data", world.config["dataset"])
logroot = os.path.join("./log", world.config["dataset"], f"denoise_{denoise_category}")

if "NNI_PLATFORM" in os.environ:
    save_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard")
    w: SummaryWriter = SummaryWriter(save_dir)
else:
    save_dir = logroot
    save_dir = join(logroot, time.strftime("%m-%d-%Hh%Mm%Ss"))
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    w: SummaryWriter = SummaryWriter(save_dir)
    logger = CompleteLogger(root=save_dir)

pprint(world.config)
w.add_text("Config", str(world.config), 0)


def write_tensorboard_metric(wirter: SummaryWriter, metric, epoch, category="Valid"):
    for key, value in metric.items():
        wirter.add_scalar(f"{category}/{key}".replace("@", "_"), value, epoch)


if __name__ == "__main__":
    # 数据处理
    data_pro = Data_Pro(dataroot, train_denoise_flag, test_denoise_flag)
    train_df, test_df = data_pro.get_data_df()
    item_num = data_pro.item_num

    batch_size = world.config["batchsize"]
    train_dataset, test_dataset = (Seq_dataset(train_df), Seq_dataset(test_df))
    train_dataloader, test_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
    )

    model = SASRec(item_num).cuda()

    state_dict_path = world.config["state_dict_path"]
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=world.config["lr"], weight_decay=world.config["weight_decay"]
    )

    start_total = time.time()
    if world.config["state_dict_path"] is None:
        patience = 0
        best_NDCG = 0
        step = 0
        best_model_dict = None
        for epoch in range(world.config["num_epochs"]):
            start_time_epoch = time.time()
            print("===================================")
            print("Start Training Epoch {}".format(epoch))

            # Train
            model.train()
            for seq_items_id in tqdm(train_dataloader):
                seq, pos, neg = utils.get_negative_items(seq_items_id, item_num)
                pos_logits, neg_logits, _ = model(seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device="cuda"), torch.zeros(
                    neg_logits.shape, device="cuda"
                )

                indices = pos != 0
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                adam_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()

                w.add_scalar(f"Train/Loss", loss, step)
                step += 1
            
            print("Time for one epoch", time.time() - start_time_epoch)

            # Test
            if epoch % 1 == 0:
                model.eval()

                t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
                write_tensorboard_metric(w, t_test, epoch, "Test")
                print("Test\n", t_test)

                if "NNI_PLATFORM" in os.environ:
                    metric = {
                        "default": t_test["NDCG@20"],
                        "ndcg@10": t_test["NDCG@10"],
                        "hit@20": t_test["HR@20"],
                        "hit@10": t_test["HR@10"],
                        "hit@50": t_test["HR@50"],
                        "ndcg@50": t_test["NDCG@50"],
                    }
                    nni.report_intermediate_result(metric)

                if t_test["NDCG@20"] > best_NDCG:
                    best_NDCG = t_test["NDCG@20"]
                    patience = 0
                    best_model_dict = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
                    print("Patience{}/10".format(patience))
                    if patience >= 10:
                        break

        if "NNI_PLATFORM" not in os.environ:
            torch.save(best_model_dict, os.path.join(save_dir, "best_model.pth"))
            model.save_item_embeddings(os.path.join(save_dir, "item_embeddings.pth"))
        model.load_state_dict(best_model_dict)

    model.eval()
    t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
    print("The Finial Test Metric for Model is following: ")
    print(t_test)

    if "NNI_PLATFORM" in os.environ:
        metric = {
            "default": t_test["NDCG@20"],
            "ndcg@10": t_test["NDCG@10"],
            "hit@20": t_test["HR@20"],
            "hit@10": t_test["HR@10"],
            "hit@50": t_test["HR@50"],
            "ndcg@50": t_test["NDCG@50"],
        }
        nni.report_final_result(metric)

    w.close()
    print("Total time:{}".format(time.time() - start_total))
    print("Training Done!")
