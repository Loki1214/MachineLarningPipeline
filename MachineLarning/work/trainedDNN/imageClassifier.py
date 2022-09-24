import torch
from torchvision import transforms
from model_definition import Net
class ImageClassifier:
	def __init__(self):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = Net().to(self.device)
		self.model.load_state_dict(torch.load('model_weights.pth'))
		self.model.eval()

	def predict(self, image):
		if isinstance(image, torch.Tensor):
			image = transforms.ToPILImage()(image)

		image_tensor = self.model.transform(image)
		if image_tensor.shape != torch.Size([self.model.image_channels,self.model.image_width,self.model.image_height]):
			print(f"Error: shape of the input image is not compatible with the neural net.\n \
				    image_tensor.shape = {image_tensor.shape}\n \
					torch.Size([1,self.model.image_width,self.model.image_height]) = {torch.Size([self.model.image_channels,self.model.image_width,self.model.image_height])}")
			exit()

		if len(image_tensor.shape) != 4:
			image_tensor = image_tensor.unsqueeze(0)

		dImage = image_tensor.to(self.device)
		pred = self.model(dImage).argmax(1)
		return int(pred[0])
