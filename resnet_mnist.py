import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

import matplotlib.pyplot as plt
from PIL import Image
from gradcam import GradCam

from torch.utils.tensorboard import SummaryWriter


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

##########################
### MNIST DATASET
##########################

data_transform = transforms.Compose([ transforms.Resize((224, 224)),
                                      transforms.ToTensor()])


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=data_transform,
                               download=False)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=data_transform)


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)


##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model


#torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES)
freeze = True
if freeze:
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 10: # Freeze 1-9 layers
            for param in child.parameters():
                param.requires_grad = False
    print('Freezed layers!')

    for param in model.layer4[1].conv2.parameters():
        param.requires_grad = True
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

grad_cam= GradCam(model=model, feature_module=model.layer4, \
                  target_layer_names=["1"], use_cuda=True)

sigmoid = nn.Sigmoid()
bce = nn.BCELoss()


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def process_explanations_train(expl, n_batches):
    tmp_list = []
    for _ in range(n_batches):
        tmp_list.append(torch.from_numpy(expl/255))
    list_of_tensors = torch.stack(tmp_list).double()
    return list_of_tensors


def calculate_measures(gts, masks):
    final_prec = 0
    final_rec = 0
    final_corr = 0

    for mask, gt in zip(masks, gts): 
        precision = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum((1-gt)*mask)) 
        final_prec = final_prec + precision

        recall = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum(gt*(1-mask)))
        final_rec = final_rec + recall 

        correlation = (1 / (gt.shape[0]*gt.shape[1])) * np.sum(gt*mask)
        final_corr = final_corr + correlation

    return final_prec, final_rec, final_corr
    

start_time = time.time()
writer = SummaryWriter("./runs/")
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        masks = grad_cam(features, training=True)
        explanation = np.load('./spiegazione_7.npy')
        list_of_expl_tensors = process_explanations_train(explanation, masks.shape[0])
        loss_gradcam = bce(masks, list_of_expl_tensors) 
        sigm_loss_gradcam = (2*sigmoid(loss_gradcam))-1

        optimizer.zero_grad()
        sigm_loss_gradcam.backward()
        optimizer.step()        

        nump_masks = masks.cpu().detach().numpy()
        list_of_expl_npy = list_of_expl_tensors.cpu().detach().numpy()
        prec, rec, corr = calculate_measures(list_of_expl_npy, nump_masks)

        prec = prec / BATCH_SIZE
        rec = rec / BATCH_SIZE
        corr = corr / BATCH_SIZE


        if (batch_idx+1) % 50 == 0:
            writer.add_scalar("Training: Loss Gradcam", sigm_loss_gradcam.item(), str(epoch + 1)+'_'+str(batch_idx+1))
            writer.add_scalar("Training: Precision", prec, str(epoch + 1)+'_'+str(batch_idx+1))
            writer.add_scalar("Training: Recall", rec, str(epoch + 1)+'_'+str(batch_idx+1))
            writer.add_scalar("Training: Correlation", corr, str(epoch + 1)+'_'+str(batch_idx+1)) 
        
        ### LOGGING
        if not (batch_idx+1) % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, (batch_idx+1), 
                     len(train_loader), sigm_loss_gradcam))

    if (epoch+1) % 2 == 0:
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
                masks = grad_cam(images, training=True)
                explanation = np.load('./spiegazione_7.npy')
                list_of_expl_tensors = process_explanations_train(explanation, masks.shape[0])
                loss_gradcam = criterion_2(masks, list_of_expl_tensors) 
                sigm_loss_gradcam = (2*sigmoid(loss_gradcam))-1

                nump_masks = masks.cpu().detach().numpy()
                list_of_expl_npy = list_of_expl_tensors.cpu().detach().numpy()
                prec, rec, corr = calculate_measures(list_of_expl_npy, nump_masks)

                prec = prec / BATCH_SIZE
                rec = rec / BATCH_SIZE
                corr = corr / BATCH_SIZE

                if (i+1) % 11 == 0:
                    writer.add_scalar("Evaluation: Loss Gradcam", sigm_loss_gradcam.item(), str(epoch + 1)+'_'+str(i+1))
                    writer.add_scalar("Evaluation: Precision", prec, str(epoch + 1)+'_'+str(i+1))
                    writer.add_scalar("Evaluation: Recall", rec, str(epoch + 1)+'_'+str(i+1))
                    writer.add_scalar("Evaluation: Correlation", corr, str(epoch + 1)+'_'+str(i+1))   
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))