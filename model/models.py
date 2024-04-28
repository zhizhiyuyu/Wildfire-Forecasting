import torch
import torch.nn as nn
import joblib
import torch.nn.functional as fn
from pytorch_msssim import ms_ssim, MS_SSIM

from sklearn.decomposition import PCA

"""Module to load the models and train them"""


def train_model(epochs, model, data_loader, learning_rate, loss_function, device, kl_div_on=False):
    """
    Trains the model for the given number of epochs

    Args:
        epochs (int): number of epochs to train the model
        model (nn.Module): model to be trained
        data_loader (torch.utils.data.DataLoader): data loader for the training data
        learning_rate (float): learning rate for the optimizer
        loss_function (nn.Module): loss function to be used
        device (str): device to be used for training
        kl_div_on (bool): whether to use KL divergence or not

    Returns:
        losses (list): list of losses for each epoch
    """


    # assert isinstance(
    #     loss_function, nn.modules.loss._Loss), 'Loss function not valid'
    # assert isinstance(
    #     data_loader, torch.utils.data.DataLoader), 'Data loader not valid'
    # assert isinstance(model, nn.Module), 'Model not valid'

    # If PCA is used, the model is trained using the fit method
    if isinstance(model, PCA):
        model = model.fit(data_loader.reshape(data_loader.shape[0], -1))
        return []

    # Else the model is trained using the train method
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses = []

        for epoch in range(epochs):
            train_loss = 0
            for (inputs, targets) in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Clearr gradients
                optimizer.zero_grad()

                # If KL divergence is used, the model returns the reconstruction loss and the KL divergence
                if kl_div_on:
                    recon_images, KL = model(inputs, targets)
                    loss = ((recon_images - targets)**2).sum() + KL
                # Else the model returns only the reconstruction loss
                else:
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                # Backpropagate the loss
                loss.backward()
                # Update the weights
                optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(data_loader)
            losses.append(train_loss)
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                  1, epochs, train_loss))

        return losses


def load_loss_function(loss_name):
    """
    Returns the loss function based on the loss name

    Args:
        loss_name (str): name of the loss function to be used
    Returns:
        loss_fn (nn.Module): loss function to be used
    """
    if loss_name == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss_name == 'L1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'BCE':
        loss_fn = nn.BCELoss()
    elif loss_name == 'MSSSIM':
        loss_fn = perceptual_loss
    else:
        raise NotImplementedError(
            'Loss function {} not implemented'.format(loss_name))
    return loss_fn


def load_encoder(model_name, latent_dim=64, device=torch.device('cpu')):
    """
    Returns the autoencoder model based on the model name

    Args:
        model_name (str): name of the model to be used
        latent_dim (int): latent dimension of the model
        device (str): device to be used for training/inference
    Returns:
        model (nn.Module): model to be use with weights loaded
    """

    if model_name == 'PCA':
        try:
            model = joblib.load('model_pth/PCA.joblib')
            print("PCA model loaded")
        except:
            print("PCA saved model not found. Train PCA model")
            model = PCA(n_components=latent_dim)
    elif model_name == 'Linear':
        model = LinearAE(latent_dim)
        weights = 'model_pth/LinearAE.pth'
    elif model_name == 'CVAE':
        model = VariationalAutoencoder(dims_latent=latent_dim)
        weights = 'model_pth/cvae_trained_model_10_epochs.pth'
    elif model_name == 'CAE1':
        model = ConvolutionalAE1(latent_dim)
        weights = 'model_pth/ConvolutionalAE1.pth'
    elif model_name == 'CAE2' or model_name == 'best':
        model = ConvolutionalAE2(latent_dim)
        weights = 'model_pth/ConvolutionalAE2.pth'
    else:
        raise NotImplementedError(
            'Encoder model {} not implemented'.format(model_name))

    try:
        print(f"Model {model_name} loaded")
        if model_name != 'PCA':
            model.load_state_dict(torch.load(weights, map_location=device))
            print("Model weights loaded. Training not required.")
            print("You can now use the model for encoding-decoding.")
        else:
            return model
    except:
        print('Failed to load the model weights. Check the model_pth folder or train the model again')
    model = model.to(device)
    return model


def load_predictor(model_name, model_encoder, input_size, output_size, hidden_size=128, num_layers=2, device=torch.device('cpu')):
    """
    Returns a lstm model based on the model name

    Args:
        model_name (str): name of the model to be used
        input_size (int): input size of the model
        hidden_size (int): hidden size of the model
        num_layers (int): number of layers in the model
        output_size (int): output size of the model
        device (str): device to be used for training/inference
    Returns:
        model (nn.Module): model to be use with weights loaded
    """

    if model_name == 'LSTM0':
        model = LSTM0(input_size, hidden_size, num_layers, output_size)
        weights = 'model_pth/LSTM0.pth'

    elif model_name == 'LSTM1' or model_name == 'best':
        model = LSTM1(input_size, hidden_size, num_layers, output_size)
        if isinstance(model_encoder, ConvolutionalAE2):
            weights = 'model_pth/LSTM1_CONVAE2.pth'
        else:
            weights = 'model_pth/LSTM1_PCA.pth'
    else:
        raise NotImplementedError(
            'LSTM Model {} not implemented'.format(model_name))

    try:
        print(f"Model {model_name} loaded")
        model.load_state_dict(torch.load(weights, map_location=device))
        print("Model weights loaded. Training not required.")
        print("You can now use the model for prediction.")
    except:
        print('Failed to load the model weights. Check the model_pth folder or train the model again')
    model.to(device)
    return model


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
        x = fn.relu(self.conv1(x))
        x = fn.relu(self.conv2(x))
        x = fn.relu(self.conv3(x))
        x = fn.relu(self.conv4(x))
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
        self.convT4 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1)

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
        z = fn.relu(self.convT4(z))
        z = fn.relu(self.convT3(z))
        z = fn.relu(self.convT2(z))
        # last layer before output is sigmoid, since we are using BCE as loss
        z = torch.sigmoid(self.convT1(z))
        return z

# Best variational autoencoder


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
        sigma = sigma.to(mu.device)
        z = mu + sigma * self.distribution.sample(mu.shape).to(mu.device)
        kl_div = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl_div

    def forward(self, x, prev_x):
        x = self.encoder(x)
        z, kl_div = self.vae_latent_space(x)
        z = self.activationOut(self.latentOut(z))
        return self.decoder(z), kl_div

    def describe(self):
        """Analysis of the model"""
        print('Number of parameters: {}'.format(
            sum(p.numel() for p in self.parameters())))
        print('Number of trainable parameters: {}'.format(sum(p.numel()
              for p in self.parameters() if p.requires_grad)))
        print('Latent dimension: {}'.format(self.latent_dim))
        print('Encoder architecture: {}'.format(self.encoder))
        print('Decoder architecture: {}'.format(self.decoder))
        print("Analysis: ")

    def print_model(self):
        """Prints the model architecture"""

        print(self.encoder)
        print(self.decoder)


# Best LSTM
class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Not necessary models. Just used for reference
class LSTM0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM0, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Not necessary models. Just used for reference
class LinearAE(nn.Module):
    def __init__(self, latent_dim):
        super(LinearAE, self).__init__()
        self.encoder = nn.Sequential(
            # 256*256 -> 1024
            nn.Linear(256*256, 1024),
            nn.ReLU(),
            # 1024 -> 512
            nn.Linear(1024, 512),
            nn.ReLU(),
            # 512 -> latent_dim
            nn.Linear(512, latent_dim)
        )

        self.decoder = nn.Sequential(
            # latent_dim -> 512
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            # 512 -> 1024
            nn.Linear(512, 1024),
            nn.ReLU(),
            # 1024 -> 256*256
            nn.Linear(1024, 256*256),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through the network

        Args:
            x (torch.tensor): input image

        Returns:
            torch.tensor: reconstructed image
        """
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def print_model(self):
        """Prints the model architecture"""

        print(self.encoder)
        print(self.decoder)

    def describe(self):
        """Analysis of the model"""

        print('Number of trainable parameters: ', sum(p.numel()
              for p in self.parameters() if p.requires_grad))
        print('Encoder Architecture: ', self.encoder)
        print('Decoder Architecture: ', self.decoder)
        print('Analysis: ')

# Not necessary models. Just used for reference


class ConvolutionalAE1(nn.Module):
    def __init__(self, latent_dim):
        super(ConvolutionalAE1, self).__init__()
        self.encoder = nn.Sequential(
            # 256 x 256 x 1
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            # 128 x 128 x 32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            # 64 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            # 32 x 32 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            # 15 x 15 x 256
            nn.ReLU(),
        )

        self.flatten_enc = nn.Sequential(
            # 65536
            nn.Linear(256 * 15 * 15, latent_dim),
            nn.ReLU(),
        )
        self.deflatten_dec = nn.Sequential(
            # 64
            nn.Linear(latent_dim, 256 * 15 * 15),
            # 65536
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # 15 x 15 x 256
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 32 x 32 x 128
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 64 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 128 x 128 x 32
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2),
            # 255 x 255 x 1
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1),
            # 256 x 256 x 1
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Compresses the input image into a latent vector

        Args:
            x (torch.tensor): Input image

        Returns:
            torch.tensor: latent vector representation of the input image(s)
        """

        x = self.encoder(x)
        x = x.view(x.size(0), 256 * 15 * 15)
        x = self.flatten_enc(x)
        return x

    def decode(self, x):
        """Reconstructs the input image from a latent vector

        Args:
            x (torch.tensor): latent vector

        Returns:
            torch.tensor: reconstructed image
        """

        x = self.deflatten_dec(x)
        x = x.view(x.size(0), 256, 15, 15)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Forward pass through the network

        Args:
            x (torch.tensor): input image

        Returns:
            torch.tensor: reconstructed image
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def print_model(self):
        """Prints the model architecture"""

        print(self.encoder)
        print(self.decoder)

    def describe(self):
        """Analysis of the model"""
        print('Number of trainable parameters: ', sum(p.numel()
              for p in self.parameters() if p.requires_grad))
        print('Encoder Architecture: ', self.encoder)
        print('Decoder Architecture: ', self.decoder)
        print('Analysis: ')


# Best model
class ConvolutionalAE2(nn.Module):
    def __init__(self, latent_dim):
        super(ConvolutionalAE2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            # 256 x 256 x 32
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # 128 x 128 x 32
            nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            # 128 x 128 x 64
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            # 64 x 64 x 64
            nn.ReLU(),
        )

        self.flatten_enc = nn.Sequential(
            # 65536
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.deflatten_dec = nn.Sequential(
            # 64
            nn.Linear(latent_dim, 128),
            # nn.ReLU(),
            nn.Linear(128, 32 * 32 * 128),

        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            # 32 x 32 x 128
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            # 64 x 64 x 128
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            # 64 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            # 128 x 128 x 64
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            # 128 x 128 x 32
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            # 256 x 256 x 32
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=1),
            # 256 x 256 x 1
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Compresses the input image into a latent vector

        Args:
            x (torch.tensor): Input image

        Returns:
            torch.tensor: latent vector representation of the input image(s)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), 64 * 64 * 64)
        x = self.flatten_enc(x)
        return x

    def decode(self, x):
        """Reconstructs the input image from a latent vector

        Args:
            x (torch.tensor): latent vector

        Returns:
            torch.tensor: reconstructed image
        """
        x = self.deflatten_dec(x)
        x = x.view(x.size(0), 128, 32, 32)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Forward pass through the network

        Args:
            x (torch.tensor): input image

        Returns:
            torch.tensor: reconstructed image
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

    def print_model(self):
        """Prints the model architecture"""

        print(self.encoder)
        print(self.decoder)

    def describe(self):
        """Analysis of the model"""

        print('Number of trainable parameters: ', sum(p.numel()
              for p in self.parameters() if p.requires_grad))
        print('Encoder Architecture: ', self.encoder)
        print('Decoder Architecture: ', self.decoder)
        print('Analysis: ')


def perceptual_loss(output_image, input_image):
    """Calculates the perceptual loss between the output and input images

    Args:
        output_image (torch.tensor): output image
        input_image (torch.tensor): input image

    Returns:
        torch.tensor: perceptual loss
    """
    # Calculate the mean squared error (MSE) loss
    mse_loss = fn.mse_loss(output_image.view(-1), input_image.view(-1))
    ms_ssim_loss = 1 - ms_ssim(input_image, output_image,
                               data_range=output_image.max() - output_image.min(), size_average=True)
    # Calculate the total loss as a combination of MSE and SSIM losses
    total_loss = mse_loss + ms_ssim_loss
    return total_loss
