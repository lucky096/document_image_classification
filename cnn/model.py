from torch import nn


class CustomModel(nn.Module):
	def __init__(self, num_classes):
		super(CustomModel, self).__init__()
		self.backbone = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Flatten(),
		)
		self.fc = nn.Linear(64*128*128, num_classes)

	def forward(self, x):
		x = self.backbone(x)
		x = self.fc(x)
		return x


if __name__ == "__main__":
	model = CustomModel(5)
	print(model)
