#%%
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
    att_weights = F.softmax(scores, dim=-1)
    return torch.matmul(att_weights, value)


#Maybe some namings differ from the book, I tried to write it myself to learn it better
class AttentionHead(torch.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.quey = torch.nn.Linear(embedding_dim, head_dim)
        self.key = torch.nn.Linear(embedding_dim, head_dim)
        self.value = torch.nn.Linear(embedding_dim, head_dim)

    def forward(self, input_embedding):

        updated_embedding = scaled_dot_product_attention(
            self.quey(input_embedding),
            self.key(input_embedding),
            self.value(input_embedding)
        )

        # updated_embedding should have dim [batch, seq_len, head_dim]
        return(updated_embedding)



class MultiHeadAttention(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding dimension {embedding_dim} is not divisible by n_heads {n_heads}")
        
        head_dim = embedding_dim // n_heads

        # Had to look what moduleList does, I used a list at first
        # Is this exploiting the parallellization opportunity of multiple independent Heads ???
        self.heads = torch.nn.ModuleList(
            [AttentionHead(embedding_dim, head_dim) for _ in range(n_heads)]
        )
        # I do not get why we are applying the linear, it is not changing the simd of the array ???
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embedding):
        # Get all the heads processed
        out_heads = [head(input_embedding) for head in self.heads]
        # Concatenate the out heads
        x = torch.cat(out_heads, dim=-1)
        # Linear projection 
        x = self.out_linear(x)
        return(x)


#%%
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoConfig

    # Load config and tokenizer from model
    model_ckpt = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Process the input from text to input embeddings
    sentence = "time flies like an arrow"
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    token_emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    inputs_embeds = token_emb(inputs.input_ids)
    
    # Let's try to run the multihead attention layer
    multihead_attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads)
    attn_output = multihead_attn(inputs_embeds)
    print(attn_output.size())
    print(attn_output)


# %%
