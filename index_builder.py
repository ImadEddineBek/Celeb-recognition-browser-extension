import argparse
import random, time, sys
from configparser import ConfigParser

import boto3
from loguru import logger
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pickle
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import pickle
from embeddings import Embedder
from index import Node
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from waitress import serve

config = ConfigParser()
config.read('config.ini')
FACE_THRESHOLD = config.getfloat('main', 'face_threshold')
METHOD = config.get('main', 'method')
CUDA = config.getboolean('main', 'cuda')
DEBUG_ENV = config.getboolean('main', 'debug')

EMBEDDING_SIZE = 512
BUCKET_NAME = 'info-ret-final-project'
REMOTE_DIRECTORY_NAME = 'data'


# DEBUG_ENV = bool(os.getenv("DEBUG_ENV", False))


# PORT = int(os.getenv("PORT", 5001))
# INDEX_TYPE = os.getenv("INDEX_TYPE", 'celebs.index')


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
            embedding = embedding[0]
            R.append((embedding, y))
    kdtree = Node(K=EMBEDDING_SIZE).build_kd_tree(R)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.idx_to_class', 'wb') as idx_to_class_file:
        pickle.dump(dataset.idx_to_class, idx_to_class_file)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.index', 'wb') as index_file:
        pickle.dump(kdtree, index_file)

    with open('indexes/' + dataset_folder.split('/')[-1] + '.data', 'wb') as data_file:
        pickle.dump(R, data_file)


def build_indexes():
    downloadDirectoryFroms3(BUCKET_NAME, REMOTE_DIRECTORY_NAME)

    sub = [os.path.join(REMOTE_DIRECTORY_NAME, o) for o in os.listdir(REMOTE_DIRECTORY_NAME)
           if os.path.isdir(os.path.join(REMOTE_DIRECTORY_NAME, o))]

    for dataset_folder in sub:
        build_kd_tree(dataset_folder)


def get_index(index_type):
    if index_type == 'celebs.index':
        index_type = 'added_.index'


    with open('indexes/' + index_type, 'rb') as index_file:
        kdtree = pickle.load(index_file)

    with open('indexes/' + index_type.split('.')[0] + '.idx_to_class', 'rb') as idx_to_class_file:
        idx_to_class = pickle.load(idx_to_class_file)

    with open('indexes/' + index_type.split('.')[0] + '.data', 'rb') as data_file:
        data = pickle.load(data_file)
    logger.info(idx_to_class)
    return kdtree, idx_to_class, data


app = Flask(__name__)


@app.route("/who_brute", methods=["GET"])
def get_brute_force():
    embedding = request.args.get('embedding')
    embedding = embedding.replace('[', '')
    embedding = embedding.replace(']', '')
    embedding = np.fromstring(embedding, dtype=float, sep=', ')
    closest = 0
    dist = np.inf
    for emb, y in data:
        cur = np.linalg.norm(emb - embedding)
        if cur < dist:
            dist = cur
            closest = y
    if closest > 1 + FACE_THRESHOLD:
        logger.info("Unknown face")
        return "Unknown face Similar to idx_to_class[closest]"
    logger.info(idx_to_class[closest] + '  ' + str(dist))
    return idx_to_class[closest]


@app.route("/who_tree", methods=["GET"])
def get_name():
    embedding = request.args.get('embedding')
    embedding = embedding.replace('[', '')
    embedding = embedding.replace(']', '')
    embedding = np.fromstring(embedding, dtype=float, sep=', ')
    return idx_to_class[kdtree.get_nn(embedding, 1)[0][1]]


CORS(app)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index which can be used to get names.')
    parser.add_argument('--port', type=int, default=5000, help='port number')
    parser.add_argument('--index_type', type=str, default='celebs.index', help='type of index')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')

    args = parser.parse_args()
    PORT = args.port
    INDEX_TYPE = args.index_type
    kdtree, idx_to_class, data = get_index(INDEX_TYPE)

    # print(PORT, INDEX_TYPE)
    if not DEBUG_ENV:
        serve(app, host='0.0.0.0', port=PORT)

    else:
        app.run(debug=True, host='0.0.0.0', port=PORT)
