from typing import Optional
import random, os, numpy as np, torch

# Set the seet to be consistent, using my student ID number:
def set_seed(seed: int = 18034856, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the seed to be random, more useful for various testing:
def set_rand_seed(deterministic: bool = False) -> None:
    # For testing different cases, set the seed to something random:
    seed = random.randint(1, 10000)
    
    # Initialize the libraries with the seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
