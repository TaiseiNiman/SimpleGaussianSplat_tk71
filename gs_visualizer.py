import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np


class Visualizer:
    def __init__(self):
        plt.ion()  # インタラクティブモードON（学習をブロックしない）
        self.fig, self.ax = plt.subplots()
        self.im = None

    def update(self, tensor_img: torch.Tensor):
        # GPU→CPU→PIL→numpy
        img_pil = to_pil_image(tensor_img.detach().cpu())
        img_np = np.asarray(img_pil)

        if self.im is None:
            self.im = self.ax.imshow(img_np)
            self.ax.axis("off")
        else:
            self.im.set_data(img_np)

        self.fig.canvas.flush_events()
        plt.pause(0.001)  # ほんの少しだけGUIイベント処理
