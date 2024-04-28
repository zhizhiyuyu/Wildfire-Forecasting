import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import pathlib 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

device = 'mps'


def load_data(path):
    """
    Load data from a specified path.

    Parameters:
    - path (str): The file path to the data file.

    Returns:
    - data (ndarray): The loaded data as a NumPy array.
    """
    return np.load(pathlib.Path(path))


def make_tensor(data, device):
    """
    Convert a NumPy array to a PyTorch tensor and move it to the specified device.

    Parameters:
    - data_1D (ndarray): The input array to be converted to a tensor.
    - device (torch.device): The device to which the tensor should be moved.

    Returns:
    - tensor (torch.Tensor): The converted tensor.
    """
    return torch.tensor(data, dtype=torch.float32).to(device)
    
class CustomDataset(Dataset):
    """
    Custom dataset class for working with image data.

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
        image_sequence = self.dataset[start_index:start_index + self.batch_size]

        # Reshape the images to (batch_size, sequence_length, height, width)
        image_sequence = image_sequence.reshape(self.batch_size, *self.dataset.shape[1:])

        # Get the current and next image pairs
        current_image = image_sequence[sample_index]
        # Handle the last pair in the batch (when current image is the last image in batch)
        if sample_index == self.batch_size - 1:
            next_image = current_image
        else:
            next_image = image_sequence[sample_index + 1]

        # Convert images to torch tensors and return the pair
        current_image_tensor = torch.tensor(current_image, dtype=torch.float32).reshape(-1, 256, 256)
        next_image_tensor = torch.tensor(
            next_image, dtype=torch.float32).reshape(-1, 256, 256)

        return current_image_tensor, next_image_tensor


def make_dataloader(MyDataset, batch_size, shuffle=True):
    """
    Create a data loader for the given dataset.

    Parameters:
    - dataset (Dataset): The dataset object to create the data loader from.
    - batch_size (int): The batch size for the data loader.

    Returns:
    - dataloader (DataLoader): The created data loader.
    """
    return torch.utils.data.DataLoader(
        MyDataset, batch_size=batch_size, shuffle=shuffle)


class VAE_Encoder(nn.Module):
    def __init__(self):
        """
        VAE Encoder class.

        This class defines the architecture of the encoder in a Variational Autoencoder (VAE).
        
        """
        super(VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        # out: [32, 128, 128]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # out: [64, 64, 64]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        # out: [128, 32, 32]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # out: [256, 16, 16]
        self.fc = nn.Linear(256*16*16, 128)

    def forward(self, x):
        """
        Forward pass of the VAE Encoder.

        Args:
            x (torch.Tensor): Input tensor to the encoder.

        Returns:
            torch.Tensor: Encoded output tensor.

        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten the data
        x = self.fc(x)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self):
        """
        VAE Decoder class.

        This class defines the architecture of the decoder in a Variational Autoencoder (VAE).

        """
        super(VAE_Decoder, self).__init__()
        self.fc = nn.Linear(128, 256*16*16)
        self.convT4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        """
        Forward pass of the VAE Decoder.

        Args:
            z (torch.Tensor): Input tensor to the decoder.

        Returns:
            torch.Tensor: Decoded output tensor.

        """
        z = self.fc(z)
        z = z.view(z.size(0), 256, 16, 16)
        z = F.relu(self.convT4(z))
        z = F.relu(self.convT3(z))
        z = F.relu(self.convT2(z))
        # last layer before output is sigmoid, since we are using BCE as loss
        z = torch.sigmoid(self.convT1(z))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims_latent):
        '''
        Class combines the Encoder and the Decoder.

        Parameters:
        - dims_latent: [int] the dimension of (number of nodes in) the
        - mean-field gaussian latent variable

        Returns:
        - data (torch.Tensor): data generated by VAE model
        - kl_div (ndarray): KL loss
        '''

        super(VariationalAutoencoder, self).__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

        self.layerMu = nn.Linear(128, dims_latent)
        self.layerSig = nn.Linear(128, dims_latent)
        self.distribution = torch.distributions.Normal(0, 1)

        self.latentOut = nn.Linear(dims_latent, 128)
        self.activationOut = nn.ReLU()

    def vae_latent_space(self, x):
        mu = self.layerMu(x)
        sigma = torch.exp(self.layerSig(x))
        z = mu + sigma * self.distribution.sample(mu.shape).to(device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl_div

    def forward(self, x, prev_x):
        x = self.encoder(x)
        z, kl_div = self.vae_latent_space(x)
        z = self.activationOut(self.latentOut(z))
        return self.decoder(z), kl_div

    
def train(autoencoder, data, device, epochs, kl_div_on=True):
    """
    Train the autoencoder model using the provided data.

    Args:
        autoencoder (nn.Module): Autoencoder model to be trained.
        data (DataLoader): DataLoader containing the training data.
        device (str): Device to be used for training (e.g., 'cpu', 'cuda').
        epochs (int): Number of training epochs.
        kl_div_on (bool, optional): Whether to include the KL divergence term in the loss. Default is True.

    Returns:
        Tuple[nn.Module, list]: Trained autoencoder model and list of losses per epoch.

    """
    autoencoder = autoencoder.to(device)
    opt = torch.optim.Adam(autoencoder.parameters())
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for current_images, next_images in data:
            current_images = current_images.to(device)
            next_images = next_images.to(device)
            opt.zero_grad()
            recon_images, KL = autoencoder(current_images, next_images)
            loss = ((recon_images - next_images)**2).sum() + KL
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")
    return autoencoder, losses


def save_model(model, path):
    """
    Save the state dictionary of the model to the specified path.

    Args:
        model (nn.Module): Model to be saved.
        path (str): File path to save the model.

    Returns:
        None

    """
    torch.save(model.state_dict(), path)


def load_model(path, ModelClass, device):
    """
    Load the model from the specified path and move it to the specified device.

    Args:
        path (str): File path to load the model.
        ModelClass (nn.Module): Model class to instantiate.
        device (str): Device to move the model to (e.g., 'cpu', 'cuda').

    Returns:
        nn.Module: Loaded model.

    """
    model = ModelClass.to(device)
    model.load_state_dict(torch.load(path))
    return model


def mse(y_obs, y_pred):
    """
    Calculate the mean squared error (MSE) between observed values and predicted values.

    Args:
        y_obs (array-like): Array of observed values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: Mean squared error (MSE) between y_obs and y_pred.

    """
    return np.square(np.subtract(y_obs, y_pred)).mean()


def make_single_forecast(input, model):
    """
    Generate a single forecast using the input data and the specified model.

    Args:
        input (tuple): Single image as numpy array.
        model (nn.Module): Trained VAE model.

    Returns:
        np.ndarray: Forecasted image.

    """
    model.eval()
    im = make_single_image_dataset(input)
    current_im = im[0][0].to(device)
    next_im = im[0][1].to(device)
    current_im = current_im.unsqueeze(dim=0)
    pred, kl_div = model(current_im, next_im)
    pred = pred.cpu().detach().numpy().reshape(-1, 256, 256).squeeze()
    return pred


def make_forecast(input, model):
    """
    Generate forecasts for multiple input data using the specified model.

    Args:
        input (list): Instance of the CustomDataset class.
        model (nn.Module): Trained VAE model.

    Returns:
        np.ndarray: Array of forecasted images.

    """
    model.eval()
    predictions = np.zeros((len(input), 256, 256))
    for i in range(len(input)):
        current_im, next_im = input[i]
        current_im = current_im.to(device)
        next_im = next_im.to(device)
        current_im = current_im.unsqueeze(dim=0)
        pred, kl_div = model(current_im, next_im)
        pred = pred.cpu().detach().numpy().reshape(-1, 256, 256).squeeze()
        predictions[i] = pred
    return predictions


def make_single_image_dataset(image):
    """
    Convert a single image into a dataset suitable for model input.

    Args:
        image (np.ndarray): Single image array of shape (256, 256).

    Returns:
        CustomDataset: Dataset object containing the single image.

    """
    image = image.reshape(-1, 256, 256)
    im = CustomDataset(image, batch_size=1, sequence_length=2)
    return im

def visualise_results(nDisplay, predictions, data, ts=None, seed=None):
    """
    Visualize the results of the VAE predictions.

    Args:
        nDisplay (int): Number of samples to display.
        predictions (np.ndarray): Array of VAE predictions with shape (num_samples, 256, 256).
        data (np.ndarray): Array of original data with shape (num_samples, 256, 256).
        seed (int): Seed for random index selection.
        ts (list): Specified index selection.

    Returns:
        None

    """
    if seed is not None:
        np.random.seed(seed)
        randomIndex = np.random.randint(0, predictions.shape[0], nDisplay)
        print(randomIndex)
        ts = randomIndex
    
    fig, axes = plt.subplots(2, nDisplay, figsize=(12, 6))
    
    for i in range(nDisplay):
        ax1 = axes[0, i]
        ax2 = axes[1, i]
        
        ax1.imshow(np.reshape(data[ts[i] + 1], (256, 256)))
        ax1.set_title(f'Observation at t_{ts[i]+1}', fontsize=10)
        ax1.axis('off')
        
        im = ax2.imshow(np.reshape(predictions[ts[i]], (256, 256)))
        ax2.set_title(f'Prediction for t_{ts[i]+1}', fontsize=10)
        ax2.axis('off')
        
      
        cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar2 = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
def covariance_matrix(X):
    """
    Compute the covariance matrix of the input data.

    Args:
        X (np.ndarray): Input data with shape (n, m), where n is the number of samples and m is the number of features.

    Returns:
        np.ndarray: Covariance matrix of the input data with shape (m, m).

    """

    means = np.array([np.mean(X, axis=1)]).transpose()
    dev_matrix = X - means
    res = np.dot(dev_matrix, dev_matrix.transpose())/(X.shape[1]-1)
    return res


def update_prediction(x, K, H, y):
    """
    Update the prediction using the Kalman filter equations.

    Args:
        x (np.ndarray): State estimate at the previous time step with shape (n,).
        K (np.ndarray): Kalman gain matrix with shape (n, m), where n is the state dimension and m is the measurement dimension.
        H (np.ndarray): Observation matrix with shape (m, n), where m is the measurement dimension and n is the state dimension.
        y (np.ndarray): Measurement vector at the current time step with shape (m,).

    Returns:
        np.ndarray: Updated state estimate at the current time step with shape (n,).

    """
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def KalmanGain(B, H, R):
    """
    Compute the Kalman gain matrix.

    Args:
        B (np.ndarray): Covariance matrix of the predicted state estimate with shape (n, n), where n is the state dimension.
        H (np.ndarray): Observation matrix with shape (m, n), where m is the measurement dimension and n is the state dimension.
        R (np.ndarray): Measurement noise covariance matrix with shape (m, m), where m is the measurement dimension.

    Returns:
        np.ndarray: Kalman gain matrix with shape (n, m), where n is the state dimension and m is the measurement dimension.

    """
    tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
    res = np.dot(B, np.dot(H.transpose(), tempInv))
    return res


def data_assimilation(B, H, R, model_compr, satellite_compr):
    """
    Perform data assimilation using the Kalman filter.

    Args:
        B (np.ndarray): Background covariance matrix with shape (n, n), where n is the state dimension.
        H (np.ndarray): Observation matrix with shape (m, n), where m is the measurement dimension and n is the state dimension.
        R (np.ndarray): Measurement noise covariance matrix with shape (m, m), where m is the measurement dimension.
        model_compr (np.ndarray): Compressed model data with shape (n, t), where n is the state dimension and t is the number of time steps.
        satellite_compr (np.ndarray): Compressed satellite data with shape (m, t), where m is the measurement dimension and t is the number of time steps.

    Returns:
        np.ndarray: Updated data array after assimilation with shape (n, t), where n is the state dimension and t is the number of time steps.

    """
    K = KalmanGain(B, H, R)
    updated_data_list = []
    for i in range(len(model_compr.T)):
        updated_data = update_prediction(
            model_compr.T[i], K, H, satellite_compr.T[i])
        updated_data_list.append(updated_data)
    updated_data_array = np.array(updated_data_list)

    return updated_data_array


def reconstruct(data_compr, model):
    """
    Reconstruct the compressed data using the specified VAE model.

    Args:
        data_compr (np.ndarray): Compressed data with shape (t, n), where t is the number of time steps and n is the feature dimension.
        model (VariationalAutoencoder): VAE model used for reconstruction.

    Returns:
        np.ndarray: Reconstructed data with shape (t, n), where t is the number of time steps and n is the feature dimension.

    """
    X = make_tensor(data_compr.T, device)
    z, kl_div = model.vae_latent_space(X)
    z = model.activationOut(model.latentOut(z))
    recon = model.decoder(z)
    recon = recon.cpu().detach().numpy()
    return recon
