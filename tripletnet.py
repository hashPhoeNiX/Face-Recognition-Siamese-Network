import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, x, y, z):
        embedded_x = self.embeddingNet(x)
        embedded_y = self.embeddingNet(y)
        embedded_z = self.embeddingNet(z)

        dist_a = F.pairwise_distance(embedded_x, embedded_y)
        dist_b = F.pairwise_distance(embedded_x, embedded_z)

        return dist_a, dist_b, embedded_x, embedded_y, embedded_z


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()