import numpy as np
import torch


def data_loading(x, batch_size, context_length, device):
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets
