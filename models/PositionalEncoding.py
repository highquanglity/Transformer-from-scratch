import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Args:
            d_model: int: the number of expected features in the input
            seq_len: int: the length of the input
            dropout: float: the dropout value
        """
        super().__init__()
        self.de_model=d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create a vecotr of shape (seq_len) using torch.arange; unsqueeze to make it from (seq_len) into (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #create a vector of shape (d_model) # a^b = exp(b*ln(a)) => 10000^(2i/d_model) = exp(2i/d_model * ln(10000))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #d_model/2; neagative sign presents for the division
        #apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        #apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        #unsqueeze pe to make it from (seq_len, d_model) into (1, seq_len, d_model) = add batch dimension
        pe = pe.unsqueeze(0)
        #register buffer to save it in the model
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #add positional encoding to the input tensor, each input has different number of sequence length
        x = x + (self.pe[: , :x.size(1), :]).requires_grad_(False) # (batch, seq_len, d_model) need to slice the pe to match the input tensor
        return self.dropout(x) # apply dropout to the input tensor to avoid overfitting



if __name__ == "__main__":
    #test the model
 
    

