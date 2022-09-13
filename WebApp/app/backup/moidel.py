import json
import shutil

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torchvision import models, transforms


def get_weight():
    url = 'https://s3xxxxxxxxxxxxxx'
    try:
        res = requests.get(url)
        with open('./weight.pth', mode='wb') as f:
            shutil.copyfileobj(res.raw, f)
    except Exception as e:
        raise RuntimeError(e)


def create_net(param_file=None, device="cpu"):
    net = models.resnet34(pretrained=True)
    if param_file:
        net.load_state_dict(torch.load(
            param_file,
            map_location=torch.device(device))
        )
    net.to(device)
    return net


def load_label(label_path):
    with open(label_path) as f:
        labels = json.load(f)
    return labels


class Predictor():
    def __init__(self, label_path, param_file=None, device="cpu"):
        self.net = create_net(param_file, device="cpu")
        self.net.eval()
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.classes = load_label(label_path)

    @torch.no_grad()
    def predict(self, img_pil):
        """
        Parameters
        ----------
        img_pil : PIL.Image.Image
        """
        img_tensor = self.trans(img_pil)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        y_pred = self.net(img_tensor)
        y_pred = F.softmax(y_pred, dim=1).detach().numpy()

        y_idx = np.argmax(y_pred)

        return y_idx, self.classes[str(y_idx)][1]