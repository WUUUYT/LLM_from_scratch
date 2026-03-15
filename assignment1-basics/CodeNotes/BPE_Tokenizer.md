# Notes on Tokenizer codes
## 完整代码
```python
from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from typing import Optional

import regex as re

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```
