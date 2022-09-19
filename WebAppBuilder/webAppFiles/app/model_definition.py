#----------------------------------------------------------
# ニューラルネットワークモデルの定義
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL

class ConvertToRGBA(object):
	def __call__(self, PILImage: PIL.Image.Image) -> PIL.Image.Image:
		return PILImage.convert('RGBA')

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.image_height = 52
		self.image_width = 52
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

		kernel_size = 3
		padding = int( (kernel_size-1)/2 )
		out_channels_1 = 32
		out_channels_2 = 64
		self.conv1 = nn.Conv2d(
			in_channels=self.image_channels,
			out_channels=out_channels_1,
			kernel_size=kernel_size,
			padding=padding)
		self.conv2 = nn.Conv2d(
			in_channels=out_channels_1,
			out_channels=out_channels_2,
			kernel_size=kernel_size,
			padding=padding)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(out_channels_2 * int(self.image_height/4) * int(self.image_width/4), 128)
		self.fc2 = nn.Linear(128, 10)
		self.activation = F.relu

	def forward(self, x):
		# print(x.shape)
		x = self.conv1(x)
		x = self.activation(x)
		x = F.max_pool2d(x, 2)
		x = self.conv2(x)
		x = self.activation(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		# print(x.shape)
		x = torch.flatten(x, 1)
		# print(x.shape)
		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output