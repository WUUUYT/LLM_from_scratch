import torch
import torch.nn.functional as F


def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str = "cpu",
    eot_token: str = "<|endoftext|>",
):
    model.eval()
    eot_token_id = tokenizer._special_str_to_id.get(eot_token, None)
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            if top_p is not None and top_p < 1.0:
                probs = nucleus_sampling(probs, top_p)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            if eot_token_id is not None and next_token_id == eot_token_id:
                break
            generated_ids.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    return tokenizer.decode(generated_ids)


def nucleus_sampling(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs - sorted_probs >= p
    sorted_probs[sorted_indices_to_remove] = 0.0
    truncated_probs = torch.zeros_like(probs)
    truncated_probs.scatter_(0, sorted_indices, sorted_probs)
    truncated_probs = truncated_probs / truncated_probs.sum()
    return truncated_probs
