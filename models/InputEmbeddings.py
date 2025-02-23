import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) *sqrt(self.d_model)
    
        