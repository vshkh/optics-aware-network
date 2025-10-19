import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from src.opticsnet.utils.seed import set_seed
from src.opticsnet.utils.device import get_device
from src.opticsnet.models.base_model import SimpleConvNet
from src.opticsnet.training.trainer import train_one_epoch, evaluate, log_metrics
from src.opticsnet.constraints.noise import add_gaussian_noise
from src.opticsnet.models.wrapped import OpticsAwareNet
from src.opticsnet.utils.config import OpticsCfg, NoiseCfg, QuantCfg


def main():
    set_seed(42)
    device = get_device()
    print("Device:", device, "| CUDA name:", torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")

    # Data (MNIST, normalized)
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    
    cfg = OpticsCfg(
        noise=NoiseCfg(enabled=True,  sigma=0.02),
        quant=QuantCfg(enabled=True, bits=8)  # activations for now
    )
    model = OpticsAwareNet(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # One quick epoch just to prove life
    stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
    eval_stats = evaluate(model, test_loader, criterion, device)
    print(f"train: {stats}")
    print(f"test : {eval_stats}")

if __name__ == "__main__":
    main()
