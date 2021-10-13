from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np

def score_cam(learn, img, class_idx, n_batch = 32, display=False):
    img, = first(learn.dls.test_dl([img]))
    learn.cuda()
    img.cuda()
    
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

    with HookBwd(learn.model.layer4) as hookbwd:
        with Hook(learn.model.layer4) as hook:
            preds = learn.model.eval()(img.cuda())
            activations = hook.stored
        preds[0,class_idx].backward()
        grads = hookbwd.stored

    _, _, H, W = img.shape

    score_saliency_map = torch.zeros((1, 1, H, W))

    activations.cuda()
    # put activation maps through relu activation
    # because the values are not normalized with eq.(1) without relu.
    activations = F.relu(activations)
    activations = F.interpolate(
        activations, (H, W), mode='bilinear')
    _, C, _, _ = activations.shape

    # normalization
    act_min, _ = activations.view(1, C, -1).min(dim=2)
    act_min = act_min.view(1, C, 1, 1)
    act_max, _ = activations.view(1, C, -1).max(dim=2)
    act_max = act_max.view(1, C, 1, 1)
    denominator = torch.where(
        (act_max - act_min) != 0., act_max - act_min, torch.tensor(1.).cuda()
    )

    activations = activations / denominator

    # generate masked images and calculate class probabilities
    probs = []
    for i in range(0, C, n_batch):
        mask = activations[:, i:i+n_batch].transpose(0, 1)
        masked_x = img * mask
        preds = learn.model.eval()(masked_x)
        probs.append(F.softmax(preds, dim=1)[:, class_idx].data)

    probs = torch.stack(probs)
    weights = probs.view(1, C, 1, 1)

    # shape = > (1, 1, H, W)
    cam = (weights * activations).sum(1, keepdim=True)
    cam = F.relu(cam)
    cam -= torch.min(cam)
    cam /= torch.max(cam)

    cam = cam.squeeze()
    
    if display:
        decoded_image = TensorImage(learn.dls.train.decode((img,))[0][0])
        _,ax = plt.subplots()
        ax.set_title('score_cam')
        decoded_image.show(ctx=ax)
        ax.imshow(cam.detach().cpu(), alpha=0.5, extent=(0,224,224,0),
                      interpolation='bilinear', cmap='jet')
    
    return cam.cpu().numpy()