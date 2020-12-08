import numpy as np
import os
import _pickle as pickle

import torch
import torchvision
import cv2
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from gradcam import GradCam 

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # last convolution
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def process_explanations_train(expl, n_batches):
    tmp_list = []
    for _ in range(n_batches):
        tmp_list.append(torch.from_numpy(expl/255))
    list_of_tensors = torch.stack(tmp_list).double()
    return list_of_tensors

def process_explanations_eval(expl, n_batches):
    tmp_list = []
    for _ in range(n_batches):
        tmp_list.append(expl/255)
    return tmp_list

def save_obj(obj, name):
    with open('./obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)


def training_model(model, train_loader, bce, cross_entropy, optimizer, epoch, num_epochs, writer):
    lamb = 0.0
    prev_maximum = -100
    prev_minimum = 100
    for i, (images, labels) in enumerate(train_loader):
            masks = grad_cam(images, training=True)
            if i == 0:
                os.mkdir('./explanations/epoch_'+str(epoch))
                temp_npy = [masks[j].cpu().detach().numpy() for j in range(6)] # lista di numpy arrays
                for idx, mask in enumerate(temp_npy):
                    mask = np.float32(mask*255)
                    mask = np.uint8(np.around(mask,decimals=0))
                    th, dst = cv2.threshold(mask, 200, 225, cv2.THRESH_BINARY)
                    cv2.imwrite('./explanations/epoch_'+str(epoch)+'/expl_'+str(idx)+'_ep_'+str(epoch)+'_.png', dst)

            explanation = np.load('./spiegazione.npy')
            list_of_expl_tensors = process_explanations_train(explanation, masks.shape[0]) 
            loss_gradcam = bce(masks, list_of_expl_tensors)

            loss_gradcam = loss_gradcam.cuda()

            images = images.cuda()
            labels = labels.cuda()
            model.train()
            output = model(images)
            loss_labels = cross_entropy(output, labels)

            total_loss = (lamb*loss_labels) + ((1-lamb)*loss_gradcam)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total = labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()

            nump_masks = masks.cpu().detach().numpy()
            list_of_expl_npy = list_of_expl_tensors.cpu().detach().numpy()
            prec, rec, corr = calculate_measures(list_of_expl_npy, nump_masks)

            prec = prec / 100
            rec = rec / 100
            corr = corr / 100
  
            if (i + 1) % 100 == 0:                
                print('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, loss gradcam: {:.4f}, loss label: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, Correlation: {:.2f}%'
                     .format(epoch + 1, num_epochs, (i + 1)*100, len(train_loader)*100, total_loss.item(), loss_gradcam.item(), loss_labels.item(), (correct/total)*100, prec*100, rec*100, corr*100))

                #writer.add_scalar("Training: Total Loss", total_loss.item(), str(epoch + 1)+'_'+str(i+1))
                #writer.add_scalar("Training: Precision", prec, str(epoch + 1)+'_'+str(i+1))
                #writer.add_scalar("Training: Recall", rec, str(epoch + 1)+'_'+str(i+1))
                #writer.add_scalar("Training: Correlation", corr, str(epoch + 1)+'_'+str(i+1))

def testing_model(model, test_loader, writer, epoch):
    for i, (images, labels) in enumerate(test_loader):
        masks = grad_cam(images, training=False)
        explanation = np.load('./spiegazione.npy')
        list_of_expl_npy = process_explanations_eval(explanation, len(masks))

        labels = labels.cuda()
        images = images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        prec, rec, corr = calculate_measures(list_of_expl_npy, masks)

        prec = prec / 100
        rec = rec / 100
        corr = corr / 100

        if (i + 1) % 100 == 0:
            print('Evaluation: Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, Correlation: {:.2f}%'
                     .format((correct/total)*100, prec*100, rec*100, corr*100))            
            #writer.add_scalar("Evaluation: Label accuracy", (correct/total)*100, str(epoch + 1)+'_'+str(i+1))
            #writer.add_scalar("Evaluation: Expl precision", prec, str(epoch + 1)+'_'+str(i+1))
            #writer.add_scalar("Evaluation: Expl recall", rec, str(epoch + 1)+'_'+str(i+1))
            #writer.add_scalar("Evaluation: Expl correlation", corr, str(epoch + 1)+'_'+str(i+1))      


def calculate_measures(gts, masks):
    final_prec = 0
    final_rec = 0
    final_corr = 0

    for mask, gt in zip(masks, gts): 
        if mask.sum() == 0:
            precision = 0
            correlation = 0
            recall = 0
        else:
            precision = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum((1-gt)*mask))
            correlation = (1 / (gt.shape[0]*gt.shape[1])) * np.sum(gt*mask)
            recall = np.sum(gt*mask) / (np.sum(gt*mask) + np.sum(gt*(1-mask)))

        final_prec = final_prec + precision
        final_rec = final_rec + recall 
        final_corr = final_corr + correlation

    return final_prec, final_rec, final_corr


if __name__ == '__main__':

    writer = SummaryWriter("./runs/")
    
    num_epochs = 10
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    cuda = torch.cuda.is_available()
    if cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU...')

    model = Net() 
    if cuda:
        model = model.cuda()
        print('Loaded model on GPU')

    grad_cam= GradCam(model=model, feature_module=model.layer2, \
                      target_layer_names=["0"], use_cuda=True)

    bce = nn.BCELoss()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=trans, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    total_step = len(train_dataset)
    for epoch in range(num_epochs):
        training_model(model, train_loader, bce, cross_entropy, optimizer, epoch, num_epochs, writer)
        testing_model(model, test_loader, writer, epoch)