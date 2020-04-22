import random, time, sys

import boto3
from loguru import logger
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle
from embeddings import Embedder
from index import Node

EMBEDDING_SIZE = 512
BUCKET_NAME = 'info-ret-final-project'
REMOTE_DIRECTORY_NAME = 'data'


def collate_fn(x):
    return x[0]


def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    s3_resource = boto3.resource('s3', aws_access_key_id='AKIAJ6V3TQK4JXROIIHQ',
                                 aws_secret_access_key='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')
    bucket = s3_resource.Bucket(bucketName)
    for object in bucket.objects.filter(Prefix=remoteDirectoryName):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        bucket.download_file(object.key, object.key)


def build_kd_tree(dataset_folder):
    logger.info(dataset_folder.split('/')[-1])
    dataset = datasets.ImageFolder(dataset_folder)
    logger.info({i: c for c, i in dataset.class_to_idx.items()})
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=4)
    embder = Embedder()
    R = []
    for x, y in loader:
        embedding = embder.embed_one(x)
        if embedding is not None:
            R.append((embedding, y))
    kdtree = Node(K=EMBEDDING_SIZE).build_kd_tree(R)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.idx_to_class', 'wb') as idx_to_class_file:
        pickle.dump(dataset.idx_to_class, idx_to_class_file)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.index', 'wb') as index_file:
        pickle.dump(kdtree, index_file)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.data', 'wb') as data_file:
        pickle.dump(R, data_file)


def build_indexes():
    # downloadDirectoryFroms3(BUCKET_NAME, REMOTE_DIRECTORY_NAME)

    sub = [os.path.join(REMOTE_DIRECTORY_NAME, o) for o in os.listdir(REMOTE_DIRECTORY_NAME)
           if os.path.isdir(os.path.join(REMOTE_DIRECTORY_NAME, o))]

    for dataset_folder in sub:
        build_kd_tree(dataset_folder)


# build_indexes()


def get_index(index_type):
    with open('indexes/' + index_type, 'rb') as index_file:
        kdtree = pickle.load(index_file)

    with open('indexes/' + index_type.split('.')[0] + '.idx_to_class', 'rb') as idx_to_class_file:
        idx_to_class = pickle.load(idx_to_class_file)

    with open('indexes/' + index_type.split('.')[0] + '.data', 'rb') as data_file:
        data = pickle.load(data_file)
    logger.info(idx_to_class)
    return kdtree, idx_to_class, data


def get_name(embedding, index_type):
    kdtree, idx_to_class, _ = get_index(index_type)
    return idx_to_class[kdtree.get_nn(embedding, 1)[0][1]]


def get_brute_force(embedding, index_type):
    _, idx_to_class, data = get_index(index_type)
    closest = 0
    dist = np.inf
    for emb, y in data:
        cur = np.linalg.norm(emb - embedding)
        if cur < dist:
            dist = cur
            closest = y

    return idx_to_class[closest]
