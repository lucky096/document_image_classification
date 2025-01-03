import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import CustomModel
from dataset import CustomDataset

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":

	device = torch.device("mps" if torch.cuda.is_available() else "cpu")
	transform = transforms.Compose([
		transforms.Resize((128,128)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
	])
	
	dataset = CustomDataset(root_dir="/Users/luckysrivastava/Workspace/data/DocumentImages", transform=transform)
        
	train_size = int(0.7 * len(dataset))
	val_size = int(0.15 * len(dataset))
	test_size = len(dataset) - (train_size + val_size)
	
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
	
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
	
	model = CustomModel(num_classes=5)
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	loss_fn = nn.CrossEntropyLoss()

	# training configurations
	num_epochs = 10
	best_val_loss = float("inf")
	early_stopping_counter = 0
	patience = 3

	writer = SummaryWriter(log_dir="/Users/luckysrivastava/Workspace/Tensorboard/DocumentImages/runs/experiment_1/")
	
	for epoch in range(num_epochs):
		model.train()
		train_loss = 0.0
		correct_train = 0
		total_train = 0
		
		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_fn(outputs, targets)
			loss.backward()
			optimizer.step()
			
			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total_train += targets.size(0)
			correct_train += predicted.eq(targets).sum().item()

		train_accuracy = 100.0 * correct_train / total_train
		print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

		writer.add_scalar("Loss/train", train_loss, epoch+1)
		writer.add_scalar("Accuracy/train", train_accuracy, epoch+1)

		model.eval()
		val_loss = 0.0
		correct_val = 0
		total_val = 0

		with torch.no_grad():
			for inputs, targets in val_loader:
				inputs, targets = inputs.to(device), targets.to(device)

				outputs = model(inputs)
				loss = loss_fn(outputs, targets)

				val_loss += loss.item()
				_, predicted = outputs.max(dim=1)
				total_val += targets.size(dim=0)
				correct_val += predicted.eq(targets).sum().item()

		val_accuracy = 100.0 * correct_val / total_val
		print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

		writer.add_scalar("Loss/val", val_loss, epoch+1)
		writer.add_scalar("Accuracy/val", val_accuracy, epoch+1)

		if val_loss > best_val_loss:
			early_stopping_counter += 1
			if early_stopping_counter >= patience:
				print(f"Early stopping triggered at epoch {epoch+1} !")
				break
		else:
			early_stopping_counter = 0
			best_val_loss = val_loss
			torch.save(model.state_dict(), "/Users/luckysrivastava/Workspace/TrainedModels/DocumentImages/experiment_1/best_model_"+str(epoch+1)+".pth")
			print(f"Model saved at {epoch+1} epoch!")


	# Compute test accuracy
	model.eval()
	correct_test = 0
	total_test = 0
	# for tensorboard visualization
	all_labels = []
	all_preds = []
	all_images = []
	# for confusion matrix
	confusion_preds = []
	confusion_labels = []
	with torch.no_grad():
		for inputs, targets in test_loader:
			inputs, targets = inputs.to(device), targets.to(device)

			outputs = model(inputs)
			_, predicted = outputs.max(dim=1)
			total_test += targets.size(dim=0)
			correct_test += predicted.eq(targets).sum().item()

			# for confusion matrix => a flat list of numpy array is needed
			confusion_labels.extend(targets.cpu().numpy())
			confusion_preds.extend(predicted.cpu().numpy())

			all_labels.append(targets.cpu())
			all_preds.append(predicted.cpu())
			all_images.append(inputs.cpu())
		
	test_accuracy = 100.0 * correct_test / total_test
	print(f"Test Accuracy {test_accuracy:.2f}%")

	writer.add_scalar("Accuracy/Test", test_accuracy, epoch+1)

	# Concatenate all images, predictions, and labels
	all_images = torch.cat(all_images)[:16]  # Select the first 16 images for visualization
	all_preds = torch.cat(all_preds)[:16]
	all_labels = torch.cat(all_labels)[:16]

	# Add images to TensorBoard with predictions and labels as captions
	for i in range(len(all_images)):
		writer.add_image(
			f"Test Image {i+1}",
			all_images[i],
			dataformats='CHW',  # Specify channel-height-width format
		)
		writer.add_text(
			f"Test Image {i+1} Info",
			f"Predicted: {all_preds[i].item()}, Actual: {all_labels[i].item()}"
		)

	#plot confusion matrix
	cm = confusion_matrix(confusion_labels, confusion_preds)
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
	plt.xlabel("Predicted")
	plt.ylabel("Actual")
	plt.title("Confusion Matrix")
	plt.show()
