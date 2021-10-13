from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
    
    
def gradcam(learner, img, class_idx, layer, display=False):
    """
    Create a gradcam visualization for a given image.
    Args:
        learner (fastai.learner.Learner): Fastai Learner object
        img (): 2D-Tensor of of shape (H, W)
        class_index (int): Index of targeted class
    """
    
    img, = first(learner.dls.test_dl([img]))
    learner.cuda()
        
    class Hook():
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_func)   
        def hook_func(self, module, input, output): 
            self.stored = output.detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()

    class HookBwd():
        def __init__(self, module):
            self.hook = module.register_backward_hook(self.hook_func)   
        def hook_func(self, module, gradient_input, gradient_output): 
            self.stored = gradient_output[0].detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()
            
    with HookBwd(layer) as hookbwd:
        with Hook(layer) as hook:
            preds = learner.model.eval()(img.cuda())
            convOutputs = hook.stored
        preds[0,class_idx].backward()
        grads = hookbwd.stored
    
    # global average pooling
    weights = grads[0].mean(dim=[1,2], keepdim=True)
    # weighted sum of activation dimensions
    cam_map = (weights.cuda() * convOutputs[0]).sum(0)
    cam_map = F.relu(cam_map)
    
    if display:
        decoded_image = TensorImage(learner.dls.train.decode((img,))[0][0])
        _,ax = plt.subplots()
        ax.set_title('gradcam')
        decoded_image.show(ctx=ax)
        ax.imshow(cam_map.detach().cpu(), alpha=0.5, extent=(0,224,224,0),
                      interpolation='bilinear', cmap='jet')

    (w, h) = (224, 224)
    heatmap = cv2.resize(cam_map.cpu().numpy(), (w, h))
    return heatmap


def guided_gradcam(backprob, gradcam, display=False):
    prod = np.einsum('ijk, ij->ijk', backprob, gradcam)
    if display:
        plt.imshow(prod)
    return prod