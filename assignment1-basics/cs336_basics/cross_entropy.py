import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits.reshape(-1, logits.size(-1))  # (samples, class)
    targets = targets.reshape(-1).long()  # (samples)

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - logits_max
    log_sums_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sums_exp - target_logits
    return loss.mean()
