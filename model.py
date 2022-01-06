# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self, base_model=models.resnet18(pretrained=True), num_classes=2):
        super(Net, self).__init__()
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.model = base_model

    def forward(self, x):
        out = self.model(x)
        out = F.softmax(out, dim=1)
        return out

class VGG16BinaryNet(nn.Module):
    def __init__(self, base_model=models.vgg16(pretrained=True)):
        super(VGG16BinaryNet, self).__init__()
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out



class StyleBinaryClassification(nn.Module):
    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=2):
        super(StyleBinaryClassification, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out



def count_pred_right(label, output):
    if (label[0] > label[1] and output[0] > output[1]) or (label[0] < label[1] and output[0] < output[1]):
        return 1
    else:
        return 0


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    # print('emd_loss:', emd_loss)
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    right_num = 0

    for i in range(mini_batch_size):
        right_num += count_pred_right(p[i], q[i])
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))

    avg_batch_loss = sum(loss_vector) / mini_batch_size
    batch_acc = right_num / mini_batch_size

    return avg_batch_loss, batch_acc

