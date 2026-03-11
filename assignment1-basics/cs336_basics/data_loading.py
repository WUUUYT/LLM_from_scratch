import numpy as np
import torch


def data_loading(x, batch_size, context_length, device):
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = np.stack([x[i : i + context_length] for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    inputs = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)

    return inputs, targets
