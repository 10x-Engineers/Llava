import numpy as np
import PIL
from PIL import Image
import torch
from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)
import torch.nn.functional as F

def preprocess(image: PIL.Image.Image):        
    im_size = (378, 378)
    return Compose(
        [
            Resize(size=im_size, interpolation=InterpolationMode.BICUBIC),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )(image)


def image_preprocessing(img_path):
    image = Image.open(img_path)
    im = image
    im = preprocess(im.convert("RGB"))
    resized_image = F.interpolate(im.unsqueeze(0), size=(378, 378), mode="bilinear")
    combined_image = resized_image
    combined_image_np = combined_image.cpu().numpy()
    return combined_image_np

def layer_weights(layer):
    state_dict = torch.load(r'./model_weights.pth', map_location="cpu",  weights_only=True)
    # print(state_dict[layer].shape)
    return state_dict[layer].detach().numpy()

