import torch
from opticsnet.models.base_model import SimpleConvNet
from opticsnet.constraints.noise import add_gaussian_noise

def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(2, 1, 28, 28, device=device)
    x = add_gaussian_noise(x, sigma=0.05)
    m = SimpleConvNet().to(device)
    y = m(x)
    assert y.shape == (2, 10)
