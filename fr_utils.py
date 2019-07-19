from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import models, transforms
from torch import nn
from torch.nn import functional as F
import numpy as np
from inception_resnet_v1 import InceptionResnetV1

input = 256 * 2 * 3
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 576, 4), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(input, 512),
                                nn.PReLU(),
                                nn.Linear(512, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)

        return x

def image_loader(image_name, transform):
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss).__init__()
        # TripletLoss.__init__(self)
        self.margin = margin
    def forward(self, x, y, z, loss_average = True):
        '''
        x - Anchor
        y - Positive
        z - Negative
        '''
        dist_ap = F.pairwise_distance(x, y).pow(2)
        dist_an = F.pairwise_distance(x, z).pow(2)

        loss = (dist_ap - dist_an + self.margin)
        loss = F.relu(loss)

        return loss.mean()if loss_average else loss.sum()

class TripletNet(nn.Module):
    def __init__(self, InceptionResnetV1, margin=1.0):
        super(TripletNet, self).__init__()
        self.embeddingNet = InceptionResnetV1
        self.margin = margin

    def forward(self, x, y, z):
        embedded_x = self.embeddingNet(x)
        embedded_y = self.embeddingNet(y)
        embedded_z = self.embeddingNet(z)

        dist_ap = F.pairwise_distance(embedded_x, embedded_y)
        dist_an = F.pairwise_distance(embedded_x, embedded_z)

        loss = (dist_ap - dist_an + self.margin)
        loss = F.relu(loss)

        return loss.mean()



def BinaryLoss(embd_x, embd_y, W, b):
    dist_xy = abs((embd_x - embd_y)).sum(1)
    binary = W * dist_xy + b
    y_hat = F.log_softmax(binary, dim=0)

    return y_hat

def img_encoding(image_name, transform, embed):
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    x_train = image.unsqueeze(0)

    embedding = embed(x_train)

    return embedding

def verification(image_path, identity, db, embednet, transform=None, threshold=0.7):
    encoding = img_encoding(image_path, transform, embednet)
    dist = torch.norm(db[identity] - encoding)
    # dist = torch.subtract(db[identity], encoding)


    if dist < threshold:
        print(f"{identity} matched!")
        access = True
    else:
        print(f'{image_path} not in the database. Join the Elite team!')
        access = False

    return dist, access

def recognition(image_path, db, embednet, transform=None, threshold=0.7):
    encoding = img_encoding(image_path, transform, embednet)


    min_dist = 100
    for name, db_encode in db.items():
        dist = torch.norm(db_encode - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        print('Not in the database')
    else:
        print(f'It\'s {identity}, the distance is {min_dist}')
