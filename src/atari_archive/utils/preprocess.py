import numpy as np
from torch import tensor
from torchvision.transforms import Compose, ToPILImage,\
    Grayscale, Resize, ToTensor


def preprocess_state(img: np.ndarray) -> tensor:
    img = img[35:190, 2:158]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img == 72] = 0
    img[img != 0] = 255
    transform = Compose([ToPILImage(), Grayscale(), Resize(80), ToTensor()])
    return transform(img)
