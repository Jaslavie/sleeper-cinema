import torch

def resolve_device(device: str) -> torch.device:
    """
    Resolve the device to use for the model.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        return torch.device("cuda")
    else:
        raise ValueError(f"Invalid device: {device}")