import os
import re
import torch

from tqdm import tqdm
import numpy as np

import numpy as np

import torch.nn.functional as F
from torch.utils.data import Dataset


def CT_variable_normalization(
    input_array: np.ndarray,
    ct_minimum_value: int = -1024,
    ct_maximum_value: int = 2048,
    output_minimum_value: float = -1,
    output_maximum_value: float = 1,
    safety_copy: bool = True,
    print_info: bool = False,
):
    """
    This function takes in a numpy array of arbitrary dimensions and normalizes
    it to a range. By default between -1 and 1. First it sets a interval for the
    possible values. Given these are meant to be CTs, by default from 4k to -1k.
    """

    assert isinstance(
        input_array, np.ndarray
    ), "We only work with numpy ndarrays here ( ｡ᵘ ᵕ ᵘ ｡)"
    assert (
        ct_minimum_value < ct_maximum_value
    ), "Minimum value must be smaller than maximum value"
    assert (
        output_minimum_value < output_maximum_value
    ), "Minimum value must be smaller than maximum value"

    # copy in case input preservation is needed
    internal_array = input_array.copy() if safety_copy else input_array

    if print_info:
        print("Original minimum value", internal_array.min())
        print("Original maximum value", internal_array.max())
        peak_to_peak = internal_array.max() - internal_array.min()
        print("Original print to print internal", peak_to_peak)

    # remove outliers from input array
    internal_array[internal_array < ct_minimum_value] = ct_minimum_value
    internal_array[internal_array > ct_maximum_value] = ct_maximum_value
    peak_to_peak = ct_maximum_value - ct_minimum_value

    if print_info:
        print("New minimum value", internal_array.min())
        print("New maximum value", internal_array.max())
        print("New print to print internal", peak_to_peak)

    # normalize it between 0 and 1.
    internal_array = (internal_array - ct_minimum_value) / peak_to_peak

    # now back to the desired range:
    amplitude = output_maximum_value - output_minimum_value
    internal_array = (internal_array * amplitude) + output_minimum_value

    return internal_array.astype(np.float32)


class Simply3D(Dataset):
    def __init__(
        self,
        path_to_dataset,
        output_size=512,
        depth_size=128,
        normalization=True,
        min_CT_clamp=-1000,
        max_CT_clamp=2000,
        output_min=-1,
        output_max=1,
    ):
        self.output_size = output_size
        self.depth_size = depth_size

        self.path_to_dataset = path_to_dataset
        self.list_of_files = os.listdir(self.path_to_dataset)

        self.normalization = normalization

        self.min_CT = min_CT_clamp
        self.max_CT = max_CT_clamp

        self.output_min = output_min
        self.output_max = output_max

        self.list_of_files = sorted(
            self.list_of_files, key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        # def get_sort_key(file_name):
        #     match = re.search(r"(?:class_no_)?(\d{1,2})", file_name)
        #     if match:
        #         return int(match.group(1))
        #     return float("inf")  # Return infinity if no match is found

        # def get_class_number(filename):
        #     # Extract the class number from the filename and convert it to an integer
        #     return int(filename.split("_")[-1].split(".")[0])

        # if not invert_sort:
        #     self.list_of_files = sorted(self.list_of_files, key=get_sort_key)
        # else:
        #     self.list_of_files = sorted(self.list_of_files, key=get_class_number)

    def __getitem__(self, index) -> np.ndarray:
        # Determine whether to use the original image or its flipped version
        assert not index >= len(self.list_of_files), "Index out of bounds"

        file_name = self.list_of_files[index]
        full_path = os.path.join(self.path_to_dataset, file_name)

        image = np.load(full_path)
        tensor = torch.from_numpy(image)

        while len(tensor.shape) < 5:
            tensor = tensor.unsqueeze(0)
            if len(tensor.shape) > 5:
                raise ValueError("Tensor has more than 5 dimensions")

        resized_tensor = F.interpolate(
            tensor,
            size=(self.depth_size, self.output_size, self.output_size),
            mode="trilinear",
            align_corners=True,
        )

        resized_tensor = resized_tensor.squeeze(0)

        if self.normalization:
            resized_tensor = self._normalization(resized_tensor)

        return resized_tensor

    def _normalization(self, resized_tensor) -> torch.Tensor:
        return CT_variable_normalization(
            input_array=resized_tensor.numpy(),
            ct_minimum_value=self.min_CT,
            ct_maximum_value=self.max_CT,
            output_minimum_value=self.output_min,
            output_maximum_value=self.output_max,
        )

    def __len__(self):
        return len(self.list_of_files)


class SimplyNumpyDataset4(Dataset):
    def __init__(
        self,
        path_to_dataset,
        output_size=512,
        normalization=True,
        min_CT_clamp=-1000,
        max_CT_clamp=2000,
        output_min=-1,
        output_max=1,
        manual_flips=True,
        invert_sort=False,
    ):
        self.output_size = output_size
        self.path_to_dataset = path_to_dataset
        self.list_of_files = os.listdir(self.path_to_dataset)
        self.normalization = normalization
        self.min_CT = min_CT_clamp
        self.max_CT = max_CT_clamp
        self.output_min = output_min
        self.output_max = output_max
        self.manual_flips = manual_flips

        def get_sort_key(file_name):
            match = re.search(r"(?:class_no_)?(\d{1,2})", file_name)
            if match:
                return int(match.group(1))
            return float("inf")  # Return infinity if no match is found

        def get_class_number(filename):
            # Extract the class number from the filename and convert it to an integer
            return int(filename.split("_")[-1].split(".")[0])

        if not invert_sort:
            self.list_of_files = sorted(self.list_of_files, key=get_sort_key)
        else:
            self.list_of_files = sorted(self.list_of_files, key=get_class_number)

    def __len__(self):
        # Return double the length of the original list of files
        if self.manual_flips:
            return len(self.list_of_files)
        else:
            return len(self.list_of_files) * 2

    def __getitem__(self, index) -> np.ndarray:
        # Determine whether to use the original image or its flipped version
        is_flipped = index >= len(self.list_of_files)
        is_out_of_bounds = index >= len(self.list_of_files) * 2
        assert not is_out_of_bounds, "Index out of bounds"

        if is_flipped and not self.manual_flips:
            index -= len(self.list_of_files)

        file_name = self.list_of_files[index]
        full_path = os.path.join(self.path_to_dataset, file_name)
        image = np.load(full_path)

        try:
            if "class_no_" in file_name:
                file_class = re.search(
                    r"image_no_(\d{1,3})_class_no_(\d{1,3})", file_name
                ).group(2)
                # raise ValueError("Discontinued naming scheme")
            elif "_class_" in file_name:
                file_class = re.search(
                    r"patient_(\d{1,3})_class_(\d{1,3})", file_name
                ).group(2)
            elif any(not c.isalpha() for c in file_name.replace(".npy", "")):
                # print("File name contains no alphabet characters")
                file_name = re.sub("\.npy", "", file_name)
                file_class = file_name.split("_")[0]
                # patient_no = file_name.split("_")[1]
                # print(f"File name: {file_name}, Class: {file_class}, Patient no: {patient_no}")
                # raise ValueError("Class not found in filename")
            else:
                print("Bad file name:", file_name)
                raise ValueError("Class not found in filename")

        except AttributeError:
            raise ValueError("Class not found in filename")

        tensor = torch.from_numpy(image)

        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) == 4:
            pass
        else:
            raise ValueError("Tensor has more than 4 dimensions")

        # tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        if is_flipped:
            # Flip the image horizontally
            tensor = torch.flip(tensor, [3])

        while len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
            if len(tensor.shape) > 4:
                raise ValueError("Tensor has more than 4 dimensions")

        resized_tensor = F.interpolate(
            tensor,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=True,
        )
        resized_tensor = resized_tensor.squeeze(0)

        if self.normalization:
            resized_tensor = self._normalization(resized_tensor)

        if type(resized_tensor) == torch.Tensor:
            resized_tensor = resized_tensor.numpy()

        return resized_tensor, int(file_class)

    def _normalization(self, resized_tensor) -> torch.Tensor:
        return CT_variable_normalization(
            input_array=resized_tensor.numpy(),
            ct_minimum_value=self.min_CT,
            ct_maximum_value=self.max_CT,
            output_minimum_value=self.output_min,
            output_maximum_value=self.output_max,
        )


class SimplyNumpyDataset5(SimplyNumpyDataset4):
    def __init__(
        self,
        path_to_dataset,
        output_size=512,
        normalization=True,
        min_CT_clamp=-1000,
        max_CT_clamp=2000,
        output_min=-1,
        output_max=1,
        manual_flips=False,
        invert_sort=False,
    ):
        super().__init__(
            path_to_dataset,
            output_size,
            normalization,
            min_CT_clamp,
            max_CT_clamp,
            output_min,
            output_max,
            manual_flips,
            invert_sort,
        )

    def __getitem__(self, index) -> np.ndarray:
        # Determine whether to use the original image or its flipped version
        assert not index >= len(self.list_of_files), "Index out of bounds"

        file_name = self.list_of_files[index]
        full_path = os.path.join(self.path_to_dataset, file_name)
        image = np.load(full_path)

        try:
            match = re.search(r"image_no_(\d{1,3})_class_no_(\d{1,3})", file_name)

            if match:
                image_number = int(match.group(1))
                class_number = int(match.group(2))
                print("File:", file_name)
                print(f"Image number: {image_number}, Class number: {class_number}")
            else:
                print("No match found")
                raise ValueError("Class not found in filename")
        except AttributeError as e:
            print("Corrupted file was,", file_name)
            raise e

        tensor = torch.from_numpy(image)
        # tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        while len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
            if len(tensor.shape) > 4:
                raise ValueError("Tensor has more than 4 dimensions")

        resized_tensor = F.interpolate(
            tensor,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=True,
        )
        resized_tensor = resized_tensor.squeeze(0)

        if self.normalization:
            resized_tensor = self._normalization(resized_tensor)

        return resized_tensor, class_number, file_name

    def _normalization(self, resized_tensor) -> torch.Tensor:
        return CT_variable_normalization(
            input_array=resized_tensor.numpy(),
            ct_minimum_value=self.min_CT,
            ct_maximum_value=self.max_CT,
            output_minimum_value=self.output_min,
            output_maximum_value=self.output_max,
        )

    def __len__(self):
        return super().__len__()
