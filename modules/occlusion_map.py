from fastai import *
from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
    
#Apply a grey patch on an image
def apply_grey_patch(image, x, y, patch_size, display=False):
    patched_image = np.array(image.cpu(), copy=True)
    patched_image[:, y:y+patch_size, x:x+patch_size] = patched_image.sum()/(image.shape[0]*image.shape[1]*image.shape[2]) # (patched_image.max() - patched_image.min()) / 2
    return tensor(patched_image).cuda()

# there is a lot of weird cpu - gpu shifting to use numpy stuff which needs to be on cpu
def occlusion(learner, img, class_index, patch_size = 20, display=False):
    """
    Create an occlusion map for a given image.
    Display the image, the occlusion map and the image overlayed with the occlusion map.
    Args:
        learner (fastai.learner.Learner): Fastai Learner object
        img (): 2D-Tensor of of shape (H, W)
        class_index (int): Index of targeted class
        patch_size (int): Size of grey patches that are slided over the image to hide parts of it.
    """
    encoded_image = first(learner.dls.test_dl([img]))
    image = np.squeeze(encoded_image[0].cpu().numpy())
    image = tensor(image).cuda()
    sensitivity_map = np.zeros((image.shape[1], image.shape[2]))
    patched_images = []
    
    # Iterate the patch over the image
    for x in range(0, image.shape[1], patch_size):
        for y in range(0, image.shape[2], patch_size):
            patched_image = apply_grey_patch(image, x, y, patch_size)
            patched_images.append(patched_image)

    patched_images = [item.float() for item in patched_images]

    # Predict for each patched image
    data = torch.stack(patched_images, axis=0)
    data.cuda()
    learner.cuda()
    preds = abs(learner.model.eval()(data))
    # kind of normalize the logits. They are sometimes quite of
    for i in range(len(preds)):
        preds[i] = preds[i] + (abs((preds.max()) - (preds[i].max())))
    confidences = preds[:, class_index]
    
    x, y = 0, 0
    for confidence in confidences:
        sensitivity_map[y:y+patch_size, x:x+patch_size] = confidence.detach().cpu().numpy()
        y+=patch_size
        if y >= image.shape[1]:
            y = 0
            x+=patch_size
    
    if display:
        img = img.resize((224,224))
        fig, axes = plt.subplots(1,3,figsize=(30,5))
        axes[0].set_title('original')
        axes[0].imshow(img)
        axes[1].set_title('occlusion')
        i = axes[1].imshow(sensitivity_map, cmap="jet", alpha=0.99)
        axes[2].set_title('original overlayed with occlusion')
        axes[2].imshow(img)
        axes[2].imshow(sensitivity_map, cmap="jet", alpha=0.5)
        fig.colorbar(i, aspect=30)  
    return sensitivity_map, patched_images, preds