import pickle
import time
import numpy as np
import streamlit as st

from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets

from embeddings import Embedder

"### This service is made by BEKKOUCH Imad for the Information Retrieval course in innopolis university"

"## "


def collate_fn(x):
    return x[0]


def update_index(index_type, dataset_folder):
    # logger.info(index_type + dataset_folder)
    # return ''
    with open('indexes/' + index_type + '.index', 'rb') as index_file:
        kdtree = pickle.load(index_file)

    with open('indexes/' + index_type + '.idx_to_class', 'rb') as idx_to_class_file:
        idx_to_class = pickle.load(idx_to_class_file)

    with open('indexes/' + index_type + '.data', 'rb') as data_file:
        data = pickle.load(data_file)
    logger.info(idx_to_class)
    # downloadDirectoryFroms3(BUCKET_NAME, dataset_folder)
    prev = len(idx_to_class.keys())
    dataset = datasets.ImageFolder('data/' + dataset_folder)
    logger.info({i + prev: c for c, i in dataset.class_to_idx.items()})
    dataset.idx_to_class = {i + prev: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=1)
    embder = Embedder()
    for x, y in loader:
        embedding = embder.embed_one(x)
        if embedding is not None:
            embedding = embedding[0]
            kdtree.kd_insert_with_split((embedding, y + prev))
            data.append((embedding, y + prev))
    for i, c in dataset.idx_to_class.items():
        idx_to_class[i] = c

    logger.info(idx_to_class)

    with open('indexes/' + dataset_folder + '_.idx_to_class', 'wb') as idx_to_class_file:
        pickle.dump(idx_to_class, idx_to_class_file)

    with open('indexes/' + dataset_folder + '_.index', 'wb') as index_file:
        pickle.dump(kdtree, index_file)

    with open('indexes/' + dataset_folder + '_.data', 'wb') as data_file:
        pickle.dump(data, data_file)
    return 'done'


"# Update an index"
"## Choose the index to be updated"
index = st.selectbox('', ('celebs', 'friends'))
"## Type the name of the s3 directory name"
user_input = st.text_input("", 'added')
if st.button('add'):
    result = update_index(index, user_input)
    st.write('result: %s' % result)
