import pandas as pd
import os
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image names and kcal values.
            images_dir (str): Directory containing all the images.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.data = pd.read_csv(csv_file)  # Load the CSV data
        self.images_dir = images_dir      # Path to the image folder
        self.transform = transform        # Optional transformations for the images
        self.valid_data = self._filter_valid_files()

    def _filter_valid_files(self):
        """
        Filters out rows with missing or invalid image files.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only valid rows.
        """
        valid_rows = []
        for _, row in self.data.iterrows():
            img_path = os.path.join(self.images_dir, row['image'])
            if os.path.exists(img_path):
                try:
                    # Verify if the file is a valid image
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_rows.append(row)
                except (UnidentifiedImageError, FileNotFoundError):
                    print(f"Skipping invalid image file: {img_path}")
            else:
                print(f"Missing file: {img_path}")
        return pd.DataFrame(valid_rows)

    def __len__(self):
        """
        Returns:
            int: Total number of valid items in the dataset.
        """
        return len(self.valid_data)

    def __getitem__(self, idx):
        """
        Retrieves the image and kcal value for a given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and kcal value.
        """
        row = self.valid_data.iloc[idx]
        file_name = row['image']
        img_path = os.path.join(self.images_dir, file_name)

        try:
            image = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        except (FileNotFoundError, UnidentifiedImageError):
            raise RuntimeError(f"Unexpected missing or invalid file during access: {img_path}")

        label = torch.tensor(row['kcal'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
