import numpy as np
from . import models
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

""" Module for loading datasets and generating torch.utils.data.Dataloader objects for training different models """


class VariationalDataset(Dataset):
    """
    Dataset for variational auto encoder.

    Args:
    - dataset (ndarray): The input data as a NumPy array.
    - batch_size (int): The size of each batch (default: 100).
    - sequence_length (int): The length of the image sequence (default: 2).

    Returns:
    - current_image (ndarray): The current image at index `idx`.
    - next_image (ndarray): The next image at index `idx + 1`.
    """

    def __init__(self, dataset, batch_size=100, sequence_length=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # Calculate the total number of batches
        self.num_batches = len(dataset) // batch_size
        self.total_samples = self.num_batches * batch_size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        # Calculate the batch index and sample index within the batch
        batch_index = index // self.batch_size
        sample_index = index % self.batch_size
        # Calculate the start index of the batch
        start_index = batch_index * self.batch_size
        # Get the sequence of images for the current batch
        image_sequence = self.dataset[start_index:start_index +
                                      self.batch_size]
        # Reshape the images to (batch_size, sequence_length, height, width)
        image_sequence = image_sequence.reshape(
            self.batch_size, *self.dataset.shape[1:])
        # Get the current and next image pairs
        current_image = image_sequence[sample_index]
        # Handle the last pair in the batch (when current image is the last image in batch)
        if sample_index == self.batch_size - 1:
            next_image = current_image
        else:
            next_image = image_sequence[sample_index + 1]

        # Convert images to torch tensors and return the pair
        current_image_tensor = torch.tensor(
            current_image, dtype=torch.float32).reshape(-1, 256, 256)
        next_image_tensor = torch.tensor(
            next_image, dtype=torch.float32).reshape(-1, 256, 256)

        return current_image_tensor, next_image_tensor


class ConvolutionalDataset(Dataset):
    def __init__(self, data, transform=None):
        """Dataset for the convolutional autoencoder

        Args:
            data (numpy.ndarray): fire dataset
            transform (torcvision.transforms, optional): Transforms the dataset. Defaults to None.
        """

        self.data = torch.tensor(data, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X_img = self.data[index]
        if self.transform:
            X_img = self.transform(X_img)
        return X_img, X_img


class LatentDataset(Dataset):
    def __init__(self, data, transform=None, timesteps=10):
        """Dataset for LSTM models

        Args:
            data (numpy.ndarray): latent space data from compressing the training dataset
            transform (torcvision.transforms, optional): Transforms the dataset. Defaults to None.
            timesteps (int, optional): Timesteps for prediction. Defaults to 10.
        """
        # Split the data into chunks of 100. Because 100 is the number of timesteps for a fire event
        data_chunks = np.split(data, data.shape[0]/100, axis=0)
        X_train = []
        y_train = []
        # For each chunk, split the data into X and y
        for data in data_chunks:
            X_train.append(data[:-timesteps])
            y_train.append(data[timesteps:])
        # Reshape the data into 2D arrays
        X_train = np.array(X_train).reshape(-1, data.shape[-1])
        y_train = np.array(y_train).reshape(-1, data.shape[-1])

        # Convert the data into tensors
        self.X_images = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        self.y_images = torch.tensor(y_train, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X_images)

    def __getitem__(self, index):
        # Get the image and label at the given index
        X_img = self.X_images[index]
        y_label = self.y_images[index]
        return X_img, y_label


def normalize_transform(sample):
    """ Normalizes the sample to have a mean of 0.5 and standard deviation of 0.5

    Args:
        sample (torch.Tensor): The sample to normalize

    Returns:
        torch.Tensor: The normalized sample
    """
    mean = [0.5, 0.5, 0.5]  # Replace with the mean values for each channel
    # Replace with the standard deviation values for each channel
    std = [0.5, 0.5, 0.5]
    transformed_sample = transforms.functional.normalize(sample, mean, std)
    return transformed_sample


def load_data(dataset_path='train', model=None, timesteps=10, batch_size=32, shuffle=False, normalise=False):
    """ Creates a torch.utils.data.Dataset based on the model provided.

    Args:
        dataset_path (str, optional): _description_. Defaults to 'train'.
        model (_type_, optional): _description_. Defaults to None.
        timesteps (int, optional): _description_. Defaults to 10.
        batch_size (int, optional): _description_. Defaults to 32.
        shuffle (bool, optional): _description_. Defaults to False.
        normalise (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if (isinstance(dataset_path, str)):
        # If a path is provided, load the data from the path
        dataset_path = pathlib.Path(dataset_path)
        try:
            data = np.load(dataset_path)
            if (data.shape[1] != 256 or data.shape[2] != 256):
                print(
                    "Expected shape (batch, 256, 256), but received shape: ", data.shape)
                raise ValueError('Data shape is not correct')
            else:
                # reshape the data to (batch, channels, height, width)
                data = data.reshape(-1, 1, 256, 256)
                # convert to torch.tensor
                # data = torch.tensor(data, dtype=torch.float32)

        except FileNotFoundError:
            print("Failed to load. Give correct path")
            raise FileNotFoundError("Failed to load. Give correct path")

    # If a numpy array is provided, load the data from the array
    elif (isinstance(dataset_path, np.ndarray)):
        data = dataset_path
        # data = torch.tensor(dataset_path, dtype=torch.float32)

    # If neither a path nor a numpy array is provided, raise an error
    else:
        raise ValueError(
            'Invalid path/data provided. Provide data path or Numpy array')

    # Creates a Custom dataset object based on the torch.nn model

    # If the model is a convolutional autoencoder load the ConvolutionalDataset
    if (isinstance(model, models.ConvolutionalAE1) or isinstance(model, models.ConvolutionalAE2)):
        dataset = ConvolutionalDataset(data)

    # If the model is an LSTM load the LatentDataset
    elif ((isinstance(model, models.LSTM0) or (isinstance(model, models.LSTM1)))):
        dataset = LatentDataset(data, timesteps=timesteps)

    # If the model is a Variational Autoencoder load the VariationalDataset
    elif(isinstance(model, models.VariationalAutoencoder)):
        dataset = VariationalDataset(data)

    elif(isinstance(model, PCA)):
        return data

        # If model has not been implemented, raise an error
    else:
        raise NotImplementedError(
            'Dataloader for this {} not implemented'.format(model))

    if normalise:
        # Apply normalization to the dataset
        dataset.transform = normalize_transform
    print(f"Data loaded successfully")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
