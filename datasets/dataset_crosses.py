import torch
from torch.utils.data import Dataset
import random
import numpy as np


def draw_x(size=16, angle=np.pi / 4):
    """
    Draw an X in a 2D array.
    :param size: The size of the 2D array.
    :param angle: The angle to rotate the X.
    """
    # Create a 2D array
    array = np.zeros((size, size))

    # Calculate center
    center = (size // 2, size // 2)

    # Draw lines
    for i in range(-size // 2, size // 2):
        x1 = int(center[0] + i * np.cos(angle))
        y1 = int(center[1] + i * np.sin(angle))
        x2 = int(center[0] + i * np.cos(angle + np.pi / 2))
        y2 = int(center[1] + i * np.sin(angle + np.pi / 2))

        if 0 <= x1 < size and 0 <= y1 < size:
            array[x1, y1] = 1
        if 0 <= x2 < size and 0 <= y2 < size:
            array[x2, y2] = -1

    return array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create and display the X
    x_image = draw_x(angle=random.uniform(0, np.pi))
    plt.imshow(x_image, cmap="gray")
    plt.title("A random X image")
    plt.colorbar()
    plt.show()


class RandomXDataset(Dataset):
    def __init__(self, size=16, length=100):
        self.size = size  # x,y dimensions of the cube
        self.length = length  # size of the dataset

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Generate and return a random cube with an X shape.
        """
        x, y = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        square_base = draw_x(size=x, angle=random.uniform(0, np.pi))
        square = torch.from_numpy(square_base).unsqueeze(0).unsqueeze(0)
        return square


class RandomCrossDataset(Dataset):
    def __init__(self, size=16, depth=16, length=100):
        self.size = size  # x,y dimensions of the cube
        self.depth = depth  # z dimensions of the cube

        self.length = length  # size of the dataset

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Generate and return a random cube with an X shape.
        """
        x, y, z = (
            self.size
            if isinstance(self.size, tuple)
            else (self.size, self.size, self.size)
        )
        cube_base = draw_x(
            size=x, angle=random.uniform(0, np.pi)
        )  # draw a random X as a base of the cube
        cube_projection = [torch.from_numpy(cube_base) for _ in range(z)]
        cube = torch.stack(cube_projection, dim=2).unsqueeze(0)

        return cube


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset_element = RandomCrossDataset((8, 8, 8))[0]

    fig, axis = plt.subplots(1, 3, figsize=(15, 5))
    axis[0].imshow(dataset_element[0][4, :, :], vmin=-1, vmax=1, cmap="gray")
    axis[1].imshow(dataset_element[0][:, 4, :], vmin=-1, vmax=1, cmap="gray")
    axis[2].imshow(dataset_element[0][:, :, 4], vmin=-1, vmax=1, cmap="gray")

    axis[0].set_title("X projection on the X axis (yz plane)")
    axis[1].set_title("X projection on the Y axis (xz plane)")
    axis[2].set_title("X projection on the Z axis (xy plane)")

    plt.show()
