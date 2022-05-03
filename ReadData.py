#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:44:28 2022

@author: tingtingzhao
"""

from skimage.io import imread_collection

col_dir = '/Users/tingtingzhao/Documents/DirectedStudyMia/DataPaperDL2019/Evaluation 0-1h diff/1hDiff/*.jpg'
undiff_dir = '/Users/tingtingzhao/Documents/DirectedStudyMia/DataPaperDL2019/Evaluation 0-1h diff/UnDiff/*.jpg'
#creating a collection with the available images
col = imread_collection(col_dir)

# We will perform the following steps while normalizing images in PyTorch:
# Load and visualize image and plot pixel values.
# Transform image to Tensors using torchvision.transforms.ToTensor()
# Calculate mean and standard deviation (std)
# Normalize the image using torchvision.transforms.Normalize().
# Visualize normalized image.
# Calculate mean and std after normalize and verify them.
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from numpy import asarray

# define custom transform function
transform = transforms.Compose([
    transforms.ToTensor()
])
  

from skimage import io
io.imshow(col[0])
#io.imshow(col[1])
undiffCol = imread_collection(undiff_dir)
#io.imshow(undiffCol[0])
#io.imshow(undiffCol[1])


##########################################################
## Exploration for Image Transformation
###########################################################
# transform the pIL image to tensor 
# image
img_tr = transform(col[0])
  
# Convert tensor image to numpy array
img_np = np.array(img_tr)
# plot the pixel values

plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

# get tensor image
  
# calculate mean and std
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)

img_transposed = np.transpose(img_tr, (1, 2, 0))

# display the normalized image
plt.imshow(img_transposed)
plt.xticks([])
plt.yticks([])
plt.imshow(col[0])

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
  
# get normalized image
img_normalized = transform_norm(col[0])
  
# convert normalized image to numpy
# array
img_np = np.array(img_normalized)
  
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")


# transpose from shape of (3,,) to shape of (,,3)
img_transposed = np.transpose(img_normalized, (1, 2, 0))

# display the normalized image
plt.imshow(img_transposed)
plt.xticks([])
plt.yticks([])
#############################################################
## Apply the Normalization for every Image in the collection
#############################################################
colNormalized = col

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def normalizeImg(col):
    colNormalized = []
    img_tr = transform(col[0])
    # get tensor image
    # calculate mean and std
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
    # get normalized image
    img_normalized = transform_norm(col[0])
    dim = list(img_normalized.shape)
    
    output = np.zeros((len(col),dim[0] ,dim[1],  dim[2]))
    
    for i in range(len(col)):
        # convert img to Tensor
        img_tr = transform(col[i])
        # get tensor image
        # calculate mean and std
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
        # get normalized image
        img_normalized = transform_norm(col[i])
        img_array = asarray(img_normalized)
        output[i,:, :, :] = img_array
     
    return output

colNormalized = normalizeImg(col)
undiffNormalized = normalizeImg(undiffCol)

# Show image to check the correctness of the code
img_transposed = np.transpose(np.array(colNormalized[0]), (1, 2, 0))
# display the normalized image
plt.imshow(img_transposed)
plt.xticks([])
plt.yticks([])

img_diff_transposed = np.transpose(np.array(undiffNormalized[0]), (1, 2, 0))
# display the normalized image
plt.imshow(img_diff_transposed)
plt.xticks([])
plt.yticks([])

diff = colNormalized
undiff = undiffNormalized

diffSubset = diff[0:(undiff.shape[0]), :]
diffSubset.shape

## Need to create labels for diff as y=1
## Need to create labels for undiff as y=0
yDiff = np.ones(diff.shape[0])
yDiff = yDiff.reshape(yDiff.shape[0], -1)
yUnDiff = np.zeros(undiff.shape[0]).reshape( undiff.shape[0], -1)
yDiffSubset = yDiff[0:undiff.shape[0]]

import numpy as np
## vertically concatenate diffSubset and undiff
X = np.vstack((diffSubset, undiff))
## vertically concatenate yDiffSubset and yUnDiff
y = np.vstack((yDiffSubset, yUnDiff))
# need to do this change since pytorch expects 0d or 1d array
y = y.flatten()

## use sklearn to split them into training and testing set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape
# (1475, 3, 480, 640)

X_test.shape
# (369, 3, 480, 640)
y.shape
# (1844, )
########################################################################




############################################################
## The next key component will be to build a pretrained 
## resnet50 pretrain model and use our training and testing 
## set to fine tune it.
## One transfer learning example we may be able to use later
## https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
## Build resnet50 as pretrained model

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True

# https://www.pluralsight.com/guides/introduction-to-resnet
# https://www.pluralsight.com/guides/image-classification-with-pytorch

## 
## Later, we need to connect our pretrained model with the new 
## layers of our model. We can use global pooling or a flatten layer to connect the dimensions of the previous layers with the new layers. With just a flatten layer and a dense layer with softmax we can perform close the model and start making classification.
## https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
#model = K.models.Sequential()
#model.add(res_model)
#model.add(K.layers.Flatten())
#model.add(K.layers.Dense(2, activation='softmax'))
# Important Pytorch tutorial

batch_size=50
from torch.utils.data import Dataset, DataLoader


## Put X_train and y labels together to DataLoader
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_data = []
for i in range(len(X_train)):
   train_data.append([X_train[i], y_train[i]])

test_data = []
for i in range(len(X_test)):
   test_data.append([X_test[i], y_test[i]])

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=50)
i1, l1 = next(iter(trainloader))
print(i1.shape)

testloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=50)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


net = models.resnet18(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)


n_epochs = 20
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
net = net.float()

# for loop for fitting the model with 20 epochs
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    # loop over all the batchs of the images and each batch has 50 images
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = net(data_)
        loss = criterion(outputs, target_)
        # Use the value of the loss function to optimize the weights
        # in the deep learning model
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(test_dataloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()
    
# The end of pytorch
##################################################################
# When using pytorch, the following code are not helpful
## If we use keras to build the model, we should use the following codes
## to preprocess the dataset
dim = col[0].shape
print(dim)
# (480, 640, 3)

## transform collection into numpy array
from numpy import asarray
import numpy as np

diff = asarray(col)
undiff = asarray(undiffCol)


diff.shape
# (1112, 480, 640, 3)
undiff.shape
# (922, 480, 640, 3)
diffSubset = diff[0:(undiff.shape[0]), :]
diffSubset.shape
# (922, 480, 640, 3)
## Need to create labels for diff as y=1
## Need to create labels for undiff as y=0
yDiff = np.ones(diff.shape[0])
yDiff = yDiff.reshape(yDiff.shape[0], -1)
yUnDiff = np.zeros(undiff.shape[0]).reshape( undiff.shape[0], -1)
yDiffSubset = yDiff[0:undiff.shape[0]]

import numpy as np
## vertically concatenate diffSubset and undiff
X = np.vstack((diffSubset, undiff))
## vertically concatenate yDiffSubset and yUnDiff
y = np.vstack((yDiffSubset, yUnDiff))

## use sklearn to split them into training and testing set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape
# (1475, 480, 640, 3)

X_test.shape
# (369, 480, 640, 3)
