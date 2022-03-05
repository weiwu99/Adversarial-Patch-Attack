# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:33:50 2021

@author: ww
"""

import numpy as np
import matplotlib.pyplot as plt


resnet_34_top1_err = np.array([0.067, 0.1204, 0.1948, 0.5801, 0.8990])
resnet_18_top1_err = np.array([0.0693, 0.1205, 0.2322, 0.3940, 0.8777])
vgg_16_top1_err = np.array([0.0600, 0.1179, 0.2236, 0.3754, 0.8334])
resnet_34_top5_err = np.array([0.0025, 0.0070, 0.0235, 0.1098, 0.4097])
resnet_18_top5_err = np.array([0.0026, 0.0057, 0.0221, 0.0948, 0.3096])
vgg_16_top5_err = np.array([0.0036, 0.0100, 0.0431, 0.0729, 0.3460])


resnet_34_top1_acc = np.array([0.0099, 0.0514, 0.8373, 0.9999])
resnet_18_top1_acc = np.array([0.0089, 0.1004, 0.5969, 0.9915])
vgg_16_top1_acc = np.array([0.0088, 0.0570, 0.1098, 0.9611])
resnet_34_top5_acc = np.array([0.1927, 0.4734, 0.9885, 1.0000])
resnet_18_top5_acc = np.array([0.2411, 0.6690, 0.9275, 1.0000])
vgg_16_top5_acc = np.array([0.3189, 0.5008, 0.3575, 0.9972])

patch_sizes = np.array([2,4,8,16])
patch_size_err = np.array([0,2,4,8,16])


plt.figure(1,figsize=(8,8))
plt.clf()
plt.plot(patch_size_err, resnet_34_top1_err, label = "Baseline ResNet-34 Top 1")
plt.plot(patch_size_err, resnet_18_top1_err, label = "ResNet-18 Top 1")
plt.plot(patch_size_err, vgg_16_top1_err, label = "VGG-16 Top 1")

plt.plot(patch_size_err, resnet_34_top5_err, label = "Baseline ResNet-34 Top 5")
plt.plot(patch_size_err, resnet_18_top5_err, label = "ResNet-18 Top 5")
plt.plot(patch_size_err, vgg_16_top5_err, label = "VGG-16 Top 5")

plt.xlabel('Patch Size', size = 16)
plt.ylabel('Error', size = 16)
plt.title('Top 1 and Top 5 Error for Different Models', size = 20)
plt.legend()
plt.show()

plt.figure(2,figsize=(8,8))
plt.clf()
plt.plot(patch_sizes, resnet_34_top1_acc, label = "Baseline ResNet-34 Top 1")
plt.plot(patch_sizes, resnet_18_top1_acc, label = "ResNet-18 Top 1")
plt.plot(patch_sizes, vgg_16_top1_acc, label = "VGG-16 Top 1")

plt.plot(patch_sizes, resnet_34_top5_acc, label = "Baseline ResNet-34 Top 5")
plt.plot(patch_sizes, resnet_18_top5_acc, label = "ResNet-18 Top 5")
plt.plot(patch_sizes, vgg_16_top5_acc, label = "VGG-16 Top 5")

plt.xlabel('Patch Size', size = 16)
plt.ylabel('Misled Accuracy', size = 16)
plt.title('Top 1 and Top 5 Accuracy for Different Models', size = 20)
plt.legend()
plt.show()