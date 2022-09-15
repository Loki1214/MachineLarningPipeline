#----------------------------------------------------------
# ニューラルネットワークモデルの定義
import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torchvision import transforms
import PIL

class ConvertToRGBA(object):
	def __call__(self, PILImage: PIL.Image.Image) -> PIL.Image.Image:
		return PILImage.convert('RGBA')

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.image_height = 50
		self.image_width = 50
		self.image_size = (self.image_height, self.image_width)
		self.image_channels = 4
		self.input_size = self.image_height * self.image_width * self.image_channels
		self.output_size = 10
		self.transform = transforms.Compose(
			[
				transforms.RandomResizedCrop(
					self.image_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)
				),
				ConvertToRGBA(),
				transforms.ToTensor()
			]
		)

		# 各クラスのインスタンス（入出力サイズなどの設定
		innerDim = 1000
		self.fc1 = nn.Linear(self.input_size, innerDim)
		self.fc2 = nn.Linear(innerDim, self.output_size)

	def forward(self, x):
		# 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
		x = self.fc1(x)
		x = torch.sigmoid(x)
		x = self.fc2(x)
		return Functional.log_softmax(x, dim=1)
