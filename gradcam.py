import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        count_layer = 0
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
                count_layer += 1
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
                count_layer += 1
            elif count_layer == 2:
                x = x.reshape(x.size(0), -1)
                x = module(x)
                count_layer += 1
            else: 
                x = module(x)
                count_layer += 1

        return target_activations, x


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_OCEAN)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.sigmoid = nn.Sigmoid()
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        #self.max_dict = dict()
        #self.min_dict = dict()
        #self.minimum = 100
        #self.maximum = -100

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, training, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=1)

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        for i, elm in enumerate(index):
            one_hot[i][elm] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        if training == True:
            grads_val = self.extractor.get_gradients()[-1].cpu()

            targets = features[-1].cpu()

            weights = torch.mean(grads_val, axis=(2, 3), keepdims=True)
            cam = torch.zeros(targets.shape[2:], dtype=torch.float32)
            cams_list = []

            for batch, pesi in enumerate(weights):
                for i, w in enumerate(pesi):
                    cam += w * targets[batch, i, :, :]
                cams_list += [cam]
                if batch < targets.shape[0]:
                    cam = torch.zeros(targets.shape[2:], dtype=torch.float32)


            another_list = []
            for i, cam in enumerate(cams_list):
                #if torch.max(cam) > self.maximum:
                #    self.maximum = torch.max(cam)
                #    maximum = self.maximum.item()
                #    self.max_dict[maximum] = 1
                #else:
                #    maximum = self.maximum.item()
                #    self.max_dict[maximum] += 1

                #if torch.min(cam) < self.minimum:
                #    self.minimum = torch.min(cam)
                #    minimum = self.minimum.item()
                #    self.min_dict[minimum] = 1
                #else:
                #    minimum = self.minimum.item()
                #    self.min_dict[minimum] += 1

                cam = torch.where(cam > 0, cam*4.38, cam*1.095)

                #cam = torch.max(cam, torch.zeros(cam.size(), dtype=torch.float32)) # ReLU
                if torch.max(cam) == 0:
                    another_list.append(cam)
                #else:
                #    cam = cam / torch.max(cam) # Normalize result between [0,1]
                #    another_list.append(cam)
                cam = self.sigmoid(cam)
                another_list.append(cam)
            #self.minimum = minimum
            #self.maximum = maximum

            masks = torch.stack(another_list).double()
            return masks #self.max_dict, self.min_dict

        if training == False:
            cams_list = []
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

            targets = features[-1]
            targets = targets.cpu().data.numpy()

            weights = np.mean(grads_val, axis=(2, 3))
            cam = np.zeros(targets.shape[1:], dtype=np.float32)

            for batch, pesi in enumerate(weights):
                for i, w in enumerate(pesi):
                    cam += w * targets[batch, i, :, :]
                cams_list += [cam]
                if batch < targets.shape[0]:
                    cam = np.zeros(targets.shape[2:], dtype=np.float32)

            another_list = []
            for cam in cams_list:
                cam = np.maximum(cam, 0) # ReLU
                if np.max(cam) == 0:
                    another_list.append(cam)
                else:
                    cam = cam / np.max(cam) # Normalize result between [0,1]
                    another_list.append(cam)

            return another_list