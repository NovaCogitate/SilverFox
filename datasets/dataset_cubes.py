import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np


class RandomCubeDataset(Dataset):
    def __init__(self, size=16, length=100):
        """
        Initialize the dataset.
        :param size: Tuple of (x, y, z) dimensions of the cube.
        :param length: The total number of items in the dataset.
        """
        self.size = size
        self.length = length

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Generate and return a random cube for each index.
        """
        x, y, z = (
            self.size
            if isinstance(self.size, tuple)
            else (self.size, self.size, self.size)
        )
        cube = torch.zeros((1, x, y, z))

        # Randomly select positions for the -1 and 1 lines
        if torch.randn(1) > 0:
            line1_pos = random.randint(0, x - 1)
            # Set the lines in the cube
            cube[0, line1_pos, :, :] = -1
        else:
            line1_pos = random.randint(0, y - 1)
            cube[0, :, line1_pos, :] = -1

        if torch.randn(1) > 0:
            line2_pos = random.randint(0, x - 1)
            # Set the lines in the cube
            cube[0, line2_pos, :, :] = -1
        else:
            line2_pos = random.randint(0, y - 1)
            cube[0, :, line2_pos, :] = -1

        # cube[0, :, line2_pos, :] = 1

        return cube


# Usage example
if __name__ == "__main__":
    IMAGE_SIZE = 16
    dataset = RandomCubeDataset(size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
    sample_cube = dataset[0]  # Get a sample cube
    print(sample_cube.shape)

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    # Plot the cube slice
    axis[0].imshow(sample_cube[0, 8, :, :], cmap="gray")
    axis[1].imshow(sample_cube[0, :, 8, :], cmap="gray")
    axis[2].imshow(sample_cube[0, :, :, 8], cmap="gray")
    plt.show()
