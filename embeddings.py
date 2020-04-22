from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os


class Embedder:
    def __init__(self, ):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        self.face_detector = MTCNN(image_size=160, margin=0, min_face_size=20,
                                   thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                                   device=device
                                   )
        self.face_embeder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def embed_one(self, img):
        x_aligned, prob = self.face_detector(img, return_prob=True)
        embeddings = None
        if x_aligned is not None:
            embeddings = self.face_embeder(torch.stack([x_aligned])).detach().cpu()[0]
        return embeddings
