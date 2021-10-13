from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np

def image_from_tensor(imagetensor):
    numpied = torch.squeeze(imagetensor)
    numpied = np.moveaxis(numpied.cpu().numpy(), 0 , -1)
    numpied = numpied - np.min(numpied)
    numpied = numpied/np.max(numpied)
    return numpied

# this callback will make all gradients positive during backprop
def clamp_gradients_hook(module, grad_in, grad_out):
    for grad in grad_in:
        torch.clamp_(grad, min=0.0)
        
# hook for guided backprop
def hooked_ReLU(m,xb,clas):
    # make sure to have the correct activation layers name
    relu_modules = [module[1] for module in m.named_modules() if str(module[1]) == "ReLU(inplace=True)" or str(module[1]) == "ReLU()"] 
    with Hooks(relu_modules, clamp_gradients_hook, is_forward=False) as _:
        preds = m(xb)
        preds[0,int(clas)].backward(retain_graph=True)
        
def guided_backprop(learn, img, y, display=False):
    img, = first(learn.dls.test_dl([img]))
    img.requires_grad=True 
    
    learn.model.cuda()
    learn.model.eval()
    
    if not img.grad is None:
        img.grad.zero_()
    hooked_ReLU(learn.model,img,y)
    img = image_from_tensor(img.grad[0].cpu())
    if display:
        plt.imshow(img)
    return img