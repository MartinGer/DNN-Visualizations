from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt

def saliency_map(learner, img, class_idx, repeats=1, noise=0.0, display=False):
    encoded_image, = first(learner.dls.test_dl([img]))
    img = encoded_image
    learner.cuda()
    # create noisy images
    if noise != 0.0:
        img = img.repeat(repeats,1,1,1)
        noise = torch.normal(mean=0, std=noise, size=(img.shape))
        noise = noise.cuda()
        img = img + noise

    # get gradients in relation to input size
    img.requires_grad=True 

    preds = learner.model.eval()(img)
    pred = preds[:,class_idx]
    pred.mean().backward()

    grads = img.grad

    # average gradients of the noisy images
    averaged_grads = torch.mean(grads, axis=0)
    # find the max of the absolute values of the gradient along each RGB channel
    dgrad_abs = torch.abs(averaged_grads)
    dgrad_max_ = torch.max(dgrad_abs, axis=0)[0]

    # normalize to range between 0 and 1
    arr_min, arr_max  = torch.min(dgrad_max_), torch.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-20)

    decoded_image = TensorImage(learner.dls.train.decode((encoded_image,))[0][0])
    decoded_image = decoded_image.cuda()

    mask = torch.unsqueeze(grad_eval, axis=0)  
    mask = mask * torch.squeeze(decoded_image/255, axis=0)

    grad_eval = grad_eval.cpu()
    
    if display:
        fig, axes = plt.subplots(1,4,figsize=(30,5))
        axes[0].set_title('original')
        decoded_image.show(ctx=axes[0])
        axes[1].set_title('saliency')
        i = axes[1].imshow(grad_eval,cmap="jet", alpha=0.99)
        axes[2].set_title('original overlayed with saliency')
        decoded_image.show(ctx=axes[2])
        axes[2].imshow(grad_eval,cmap="jet", alpha=0.75)
        axes[3].set_title('original multiplied with saliency')
        mask.show(ctx=axes[3])
        fig.colorbar(i, aspect=30)
    
    return grad_eval