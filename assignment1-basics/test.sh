cd /storage/ice1/0/5/ywu3117/LLM_from_scratch/assignment1-basics
source .venv/bin/activate

python -c "
import numpy as np

# 1. 检查数据目录中有哪些文件
import os
print('=== Data files ===')
for f in sorted(os.listdir('data')):
    size = os.path.getsize(f'data/{f}')
    print(f'  {f}: {size/1e6:.1f} MB')

# 2. 检查训练数据 token 范围
print('\n=== Train data ===')
train = np.memmap('data/owt_train_ids.uint16', dtype=np.uint16, mode='r')
print(f'Total tokens: {len(train):,}')
print(f'Token ID range: [{train.min()}, {train.max()}]')
print(f'Unique tokens (first 1M): {len(np.unique(train[:1_000_000]))}')
print(f'First 20 token IDs: {train[:20].tolist()}')

# 3. 检查验证数据（如果存在）
val_path = 'data/owt_valid_ids.uint16'
if os.path.exists(val_path):
    val = np.memmap(val_path, dtype=np.uint16, mode='r')
    print(f'\n=== Val data ===')
    print(f'Total tokens: {len(val):,}')
    print(f'Token ID range: [{val.min()}, {val.max()}]')
else:
    print(f'\n!!! {val_path} DOES NOT EXIST !!!')
    print('Available uint16 files:')
    for f in os.listdir('data'):
        if f.endswith('.uint16'):
            print(f'  {f}')

# 4. 对比 cross_entropy 与 PyTorch 内置实现
import torch
import torch.nn.functional as F
torch.manual_seed(42)
logits = torch.randn(128, 32000)
targets = torch.randint(0, 32000, (128,))
ref = F.cross_entropy(logits, targets)

logits_max = torch.max(logits, dim=-1, keepdim=True).values
shifted = logits - logits_max
log_sum = torch.log(torch.sum(torch.exp(shifted), dim=-1))
tgt_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
mine = (log_sum - tgt_logits).mean()

print(f'\n=== Cross Entropy comparison ===')
print(f'PyTorch F.cross_entropy: {ref.item():.6f}')
print(f'Your cross_entropy:      {mine.item():.6f}')
print(f'Difference:              {abs(ref.item()-mine.item()):.2e}')
"
