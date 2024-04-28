from pytest import mark
import numpy as np
import matplotlib.pyplot as plt
import torch
from . import CVAE
device = 'mps'

array = np.random.rand(4, 256, 256)
array2 = np.array([[0,2],[1,1],[2,0]]).T
array3 = np.random.rand(5,256,256)
np.save('path.npy', array)
dataset = CVAE.CustomDataset(array, batch_size=2, sequence_length=2)
dataloader = CVAE.make_dataloader(dataset, batch_size=1, shuffle=True)
autoencoder = CVAE.VariationalAutoencoder(64)

def test_load_data():
    """Test that output is a numpy array"""
    data = CVAE.np.load('path.npy')
    assert(type(data)==np.ndarray)

def test_make_tensor():
    """Check that the return type is a tensor"""
    tensor = CVAE.make_tensor(array, device)
    assert(torch.is_tensor(tensor))

def test_CustomDataset():
    """Check that the return length is the same as the array length and the return type is tensor"""
    dataset = CVAE.CustomDataset(array, batch_size=2, sequence_length=2)
    current_im, next_im = dataset[0]
    assert(len(dataset)==len(array))
    assert(torch.is_tensor(current_im))
    assert(torch.is_tensor(next_im))

def test_make_dataloader():
    """Check that the return type is a torch dataloader"""
    dataloader = CVAE.make_dataloader(dataset, batch_size=1, shuffle=True)
    assert(type(dataloader)==torch.utils.data.DataLoader)

def test_VariationalAutoencoder():
    """Check that the class instansiation of the VAE"""
    ae = CVAE.VariationalAutoencoder(64)

def test_train():
    """Check that the train function runs"""
    ae, losses = CVAE.train(autoencoder, dataloader, device, epochs=1)

def test_covariance_matrix():
    """Test that covariance is being calculated correctly using known test case"""
    res = CVAE.covariance_matrix(array2)
    assert(np.allclose(res, np.array([[1,-1],[-1,1]])))

def test_update_prediction():
    """Check that the update_prediction function returns the original array if the two arrays are equal"""
    K = np.identity(256)
    H = K
    res = CVAE.update_prediction(array[0], K, H, array[0])
    assert(np.allclose(res, array[0]))

def test_KalmanGain():
    """Check that the KalmanGain function produces the correct result"""
    nNodes = 1
    I = np.identity(nNodes)
    R = I
    H = I 
    B = I
    res = CVAE.KalmanGain(B, H, R)
    assert(res[0][0]==0.5)

def test_data_assimiliation():
    """Check that the data assimilation runs"""
    ae, losses = CVAE.train(autoencoder, dataloader, device, epochs=1)
    arr = CVAE.make_tensor(array[0], device).reshape(-1,1,256,256)
    data_compr = ae.encoder(arr)
    data_compr = data_compr.cpu().detach().numpy()
    nNodes = 1
    I = np.identity(nNodes)
    R = I
    H = I 
    B = CVAE.covariance_matrix(data_compr) 
    updated_data_array = CVAE.data_assimilation(B, H, R, data_compr, data_compr)

def test_reconstruct():
    """Check that the reconstruction of compressed data runs"""
    ae, losses = CVAE.train(autoencoder, dataloader, device, epochs=1)
    arr = CVAE.make_tensor(array[0], device).reshape(-1,1,256,256)
    data_compr = ae.encoder(arr)
    data_compr = data_compr.cpu().detach().numpy().T
    recon = CVAE.reconstruct(data_compr, ae)

def test_make_single_image_dataset():
    """Test that the correct custom dataset has been created for the input image"""
    im = CVAE.make_single_image_dataset(array[0])
    im1 = im[0][0].cpu().detach().numpy().squeeze(0)
    im2 = im[0][1].cpu().detach().numpy().squeeze(0)
    assert(np.allclose(im1, array[0]))
    assert(np.allclose(im2, array[0]))

def test_mse():
    """Check that when both arrays are identical the MSE returns 0"""
    y_obs = np.zeros((4,28,28))
    y_pred = np.zeros((4,28,28))
    mse = CVAE.mse(y_obs, y_pred)
    assert(np.allclose(mse,0))

def test_visualise_results():
    """Check that the visualise results function runs"""
    CVAE.visualise_results(4,array,array3, seed=42)

def test_make_forecast():
    """Check that the make forecast function runs and produces a numpy array"""
    ae = autoencoder.to(device)
    pred = CVAE.make_forecast(dataset, ae)
    assert(type(pred)==np.ndarray)

def test_make_single_forecast():
    """Check that the make single forecast function runs and produces a numpy array"""
    image = array[0]
    ae = autoencoder.to(device)
    pred = CVAE.make_single_forecast(image, ae)
    assert(type(pred)==np.ndarray)

def test_save_and_load_model():
    """Check that the save and load model functions run"""
    CVAE.save_model(autoencoder, 'testing_model_saving.pth')
    CVAE.load_model('testing_model_saving.pth', autoencoder, device)

if __name__ == "__main__":
    test_make_tensor()
    test_CustomDataset()
    test_make_dataloader()
    test_VariationalAutoencoder()
    test_train()
    test_reconstruct()
    test_KalmanGain()
    test_update_prediction()
    test_covariance_matrix()
    test_load_data()
    test_make_single_image_dataset()
    test_mse()
    test_visualise_results()
    test_make_forecast()
    test_make_single_forecast()
    test_save_and_load_model()