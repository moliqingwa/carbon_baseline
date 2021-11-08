import os
import sys
import logging

import csv
import torch


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir, model_name="status.pt"):
    return os.path.join(model_dir, model_name)


def get_status(model_dir, device, model_name="status.pt"):
    path = get_status_path(model_dir, model_name)
    return torch.load(path, map_location=device)


def save_status(status, model_dir, model_name="status.pt"):
    path = get_status_path(model_dir, model_name)
    create_folders_if_necessary(path)
    torch.save(status, path)


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
