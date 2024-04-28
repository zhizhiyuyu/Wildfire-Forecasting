import torch
import torch.nn as nn
# import joblib
# import torch.nn.functional as fn
from sklearn.decomposition import PCA
import torchvision.transforms as transforms

import numpy as np
from . import models
from . import dataloader

""" Module for testing the functions in the models.py and dataloader.py files"""

test_input_image = np.random.rand(5, 256, 256)
test_output_image = np.random.rand(5, 256, 256)

###Tests for models.py ######
def test_load_loss_function():
    """Tests the load_loss_function function
    """
    loss_fn = models.load_loss_function('MSE')
    assert isinstance(loss_fn, nn.modules.loss.MSELoss), 'MSE not loaded'
    loss_fn = models.load_loss_function('L1')
    assert isinstance(loss_fn, nn.modules.loss.L1Loss), 'L1 not loaded'
    loss_fn = models.load_loss_function('BCE')
    assert isinstance(loss_fn, nn.modules.loss.BCELoss), 'BCE not loaded'
    loss_fn = models.load_loss_function('MSSSIM')
    assert callable(loss_fn), 'MSSSIM not loaded'
    print('All tests passed')


def test_load_encoder():
    """Tests the load_encoder function
    """
    model = models.load_encoder('PCA')
    assert isinstance(model, PCA), 'PCA not loaded'
    model = models.load_encoder('Linear')
    assert isinstance(model, models.LinearAE), 'LinearAE not loaded'
    model = models.load_encoder('CVAE')
    assert isinstance(model, models.VariationalAutoencoder), 'ConvolutionalVAE not loaded'
    model = models.load_encoder('CAE1')
    assert isinstance(model, models.ConvolutionalAE1), 'ConvolutionalAE1 not loaded'
    model = models.load_encoder('CAE2')
    assert isinstance(model, models.ConvolutionalAE2), 'ConvolutionalAE2 not loaded'
    print('All tests passed')


# #create test function for load_predictor
def test_load_predictor():
    """Tests the load_lstm function
    """
    model = models.load_predictor('LSTM0', 1, 1, 1, 1)
    assert isinstance(model, models.LSTM0), 'LSTM0 not loaded'
    model = models.load_predictor('LSTM1', 1, 1, 1, 1)
    assert isinstance(model, models.LSTM1), 'LSTM10 not loaded'
    print('All tests passed')

###Tests for dataloader.py ######

def test_load_data():
    """
    Tests the load_data function
    """
    test_data = np.random.rand(200, 256, 256)
    np.save('test_data.npy', test_data)
    latent_dim = 64
    data = dataloader.load_data(test_data, models.ConvolutionalAE1(latent_dim))
    assert isinstance(data, torch.utils.data.DataLoader), 'Data not loaded'
    data = dataloader.load_data(test_data, models.LSTM0(1,1,1,1))
    assert isinstance(data, torch.utils.data.DataLoader), 'Data not loaded'
    data = dataloader.load_data(test_data, models.LSTM1(1,1,1,1))
    assert isinstance(data, torch.utils.data.DataLoader), 'Data not loaded'
    data = dataloader.load_data(test_data, models.VariationalAutoencoder(latent_dim))
    assert isinstance(data, torch.utils.data.DataLoader), 'Data not loaded'
    data = dataloader.load_data(test_data, model = PCA())
    assert isinstance(data, np.ndarray), 'Data not loaded'
    data = dataloader.load_data('test_data.npy', model = models.ConvolutionalAE1(latent_dim))
    print('All tests passed')

def test_normalize_transform():
    '''
    Tests the normalize_transform function
    '''
    test_data = torch.randn(3, 256, 256)
    normalized_data = dataloader.normalize_transform(test_data)
    assert isinstance(normalized_data, torch.Tensor), 'Data not normalized'
    print('All tests passed')
