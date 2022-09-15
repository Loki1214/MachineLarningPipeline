import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None, header=None, names=None):
		self.csv_file = pd.read_csv(csv_file, header=header, names=names)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.csv_file)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.root_dir,
								self.csv_file.iloc[idx, 0])
		image = Image.open(img_name)
		labels = self.csv_file.iloc[idx, 1:]
		# print(self.csv_file.iloc[idx, 1])
		# labels = np.array([labels])
		# labels = labels.astype('float')

		# sample = {'image': image, 'labels': labels}

		if self.transform:
			image = self.transform(image)

		# sample = (image, labels)
		return (image, labels.iat[0])
