from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
    
def visualize_activations(learn, image, layer):
    """
    Visualize the activations of a models layer for a given image.
    Args:
        learner (fastai.learner.Learner): Fastai Learner object
        img (): 2D-Tensor of of shape (H, W)
        layer (layer): A models layer.
    """
    img, = first(learn.dls.test_dl([image]))
    learn.cuda()
    img.cuda()
    
    class Hook():
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_func)   
        def hook_func(self, module, input, output): 
            self.stored = output.detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()

    with Hook(layer) as hook:
        preds = learn.model.eval()(img.cuda())
        activations = hook.stored

    images_per_row = 16

    activations = activations.permute(0, 2, 3, 1)
    n_features = activations.shape[-1] # Number of features in the feature map
    size = activations.shape[1] # The feature map has shape (1, size, size, n_features).
    
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = activations[0, :, :, col * images_per_row + row].cpu()
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).numpy().astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(str(layer))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    return activations