from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
        ) -> torch.Tensor : 

    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)