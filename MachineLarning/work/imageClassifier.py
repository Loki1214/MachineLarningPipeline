import torch
from torchvision import transforms
from model_definition import Net
class ImageClassifier:
	def __init__(self, image_width, image_height):
		self.image_width  = image_width
		self.image_height = image_height
		self.image_size   = image_width * image_height
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.transform = transforms.Compose([
			transforms.ToTensor()
		])
		self.model = Net(self.image_size, 10).to(self.device)
		self.model.load_state_dict(torch.load('model_weights.pth'))
		self.model.eval()

	def predict(self, image):
		if isinstance(image, torch.Tensor):
			image_tensor = image
			print("Got a torch.Tensor.")
		else:
			image_tensor = self.transform(image)
			print("Got a PIL image.")
		if image_tensor.shape != torch.Size([1,self.image_width,self.image_height]):
			print(f"Error: shape of the input image is not compatible with the neural net. image_tensor.shape = {image_tensor.shape}")
			exit()
		dImage = image_tensor.to(self.device)
		dImage = dImage.view(-1, self.image_size)
		pred = self.model(dImage).argmax(1)
		return int(pred[0])
