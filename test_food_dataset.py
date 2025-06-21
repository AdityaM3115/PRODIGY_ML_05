from food_dataset import FoodDataset
from torchvision import transforms

# Define paths
csv_path = "F:/vscode/python/task4/dataset/data.csv"  # Path to your CSV file
images_dir = "F:/vscode/python/task4/dataset/images"  # Path to your image folder

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset
dataset = FoodDataset(csv_file=csv_path, images_dir=images_dir, transform=data_transforms)

# Access the first sample
image, label = dataset[0]
print("Image shape:", image.shape)  # Should be (3, 224, 224)
print("Label (kcal):", label)
