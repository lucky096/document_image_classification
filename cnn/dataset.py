import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import  torchvision.transforms as T

""" 
# Method1 : Using torchvision.datasets.fromImageFolder
dataset = datasets.ImageFolder(root="data/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
for images, labels in dataloader:
	print(images.shape, labels.shape)
"""

# Method2: By defining a custom Dataset class
class CustomDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.data = []
		self.class_to_idx = {}

		self._load_data()
	
	def _load_data(self):
		classes = sorted(os.listdir(self.root_dir))
		self.class_to_idx = {class_name:idx for idx,class_name in enumerate(classes)}
		
		for class_name in classes:
			class_dir = os.path.join(self.root_dir, class_name)
			if not os.path.isdir(class_dir):
				continue
			for img_name in os.listdir(class_dir):
				img_path = os.path.join(class_dir, img_name)
				self.data.append((img_path, self.class_to_idx[class_name]))
	
	def __len__(self):
		return len(self.data)
  
	def __getitem__(self, idx):
		img_path, label = self.data[idx]
		image = Image.open(img_path).convert("RGB")
		
		if self.transform:
			image = self.transform(image)
		
		return image,label


if __name__ == "__main__":

	transform = T.Compose([
		T.Resize((128,128)),
		T.ToTensor(),
		T.Normalize(mean=[0.5,0.5,0.4], std=[0.5,0.5,0.5]),
	])

	dataset = CustomDataset(root_dir="/Users/luckysrivastava/Workspace/data/DocumentImages/", transform=transform)

	train_size = int(0.7 * len(dataset)) 
	val_size = int(0.15 * len(dataset))
	test_size = len(dataset) - (train_size + val_size) 

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

	print(f"Total samples: {len(dataset)}")
	print(f"Train samples: {len(train_dataset)}")
	print(f"Test samples: {len(test_dataset)}")
	 
	for image, label in train_loader:
		print(image.shape)
		print(label.shape)
		break