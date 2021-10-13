# https://www.kaggle.com/daisukelab/verifying-cnn-models-with-cam-and-etc-fast-ai
from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale


def visualize_kernel(learn, layers):
#     layers = [
#         # Those are the possible conv layers
#         list(learn.model.children())[0],

#         list(learn.model.children())[4][0].conv2,
    #     list(learn.model.children())[4][1].conv2,
    #     list(learn.model.children())[4][2].conv2,

    #     list(learn.model.children())[5][0].conv2,
    #     list(learn.model.children())[5][1].conv2,
    #     list(learn.model.children())[5][2].conv2,
    #     list(learn.model.children())[5][3].conv2,

    #     list(learn.model.children())[6][0].conv2,
    #     list(learn.model.children())[6][1].conv2,
    #     list(learn.model.children())[6][2].conv2,
    #     list(learn.model.children())[6][3].conv2,
    #     list(learn.model.children())[6][4].conv2,
    #     list(learn.model.children())[6][5].conv2,

    #     list(learn.model.children())[7][0].conv2,
    #     list(learn.model.children())[7][1].conv2,
    #     list(learn.model.children())[7][2].conv2,
#     ]
    
    # for attention use something like list(learn.model.children())[0][0]
    
    for layer in layers:
        weights = layer.weight.data.cpu().numpy()
        weights_shape = weights.shape 
        weights = minmax_scale(weights.ravel()).reshape(weights_shape)
        fig, axes = plt.subplots(8, 8, figsize=(6,6))
        weights = weights[:,:3,:,:] # take only three channels
        for i, ax in enumerate(axes.flat):
            weight = np.rollaxis(weights[i], 0, 3)
            ax.imshow(weight)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)