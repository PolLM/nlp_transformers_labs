#%%
from config import Config

import torch as pt
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask = None): 

    dim_k = query.size(-1)
    scores = pt.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    att_weights = F.softmax(scores, dim=-1)
    return pt.matmul(att_weights, value)


class SelfAttentionHead(pt.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.query = pt.nn.Linear(embedding_dim, head_dim)
        self.key = pt.nn.Linear(embedding_dim, head_dim)
        self.value = pt.nn.Linear(embedding_dim, head_dim)

    def forward(self, input_embedding, return_kv=False):

        q = self.query(input_embedding)
        k = self.key(input_embedding)
        v = self.value(input_embedding)

        updated_embedding = scaled_dot_product_attention(q, k, v)

        # updated_embedding should have dim [batch, seq_len, head_dim]
        if not return_kv:
            return(updated_embedding)
        else:
            return(updated_embedding, k, v)



class MultiHeadSelfAttention(pt.nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.embedding_dim % config.num_self_attn_heads != 0:
            raise ValueError(f"embedding dimension {config.embedding_dim} is not divisible by num_heads {num_heads}")
        
        head_dim = config.embedding_dim // config.num_self_attn_heads

        # Had to look what moduleList does, I used a list at first
        # Is this exploiting the parallellization opportunity of multiple independent Heads ???
        self.heads = pt.nn.ModuleList(
            [SelfAttentionHead(config.embedding_dim, head_dim) for _ in range(config.num_self_attn_heads)]
        )
        # I do not get why we are applying the linear, it is not changing the simd of the array ???
        self.out_linear = pt.nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, input_embedding, return_kv=False):
        # Get all the heads processed


        if not return_kv:
            out_emb = [head(input_embedding, return_kv) for head in self.heads]
        else:
            out_emb, out_k, out_v = [], [], []
            for head in self.heads:
                out_emb_i, out_k_i, out_v_i = head(input_embedding, return_kv)
                out_emb.append(out_emb_i)
                out_k.append(out_k_i.unsqueeze(2))
                out_v.append(out_v_i.unsqueeze(2))
            # out_k_i has shape (batch, seq_len, 1, head_dim)
            # will create a tensor of shape (batch, seq_len, num_heads, head_dim)
            # will use it if we need to store the kv cahce on the encoder side
            out_k = pt.cat(out_k, dim=2)
            out_v = pt.cat(out_v, dim=2)

        # Concatenate the out heads
        x = pt.cat(out_emb, dim=-1)
        # Linear projection 
        x = self.out_linear(x)

        if  not return_kv:
            return(x)
        else:
            return(x, out_k, out_v)
        

# 1d convolution / position-wise feed forward 2 layer NN
class FeedForward(pt.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer1 = pt.nn.Linear(config.embedding_dim, config.ff_latent_dim)
        self.layer2 = pt.nn.Linear(config.ff_latent_dim, config.embedding_dim)
        self.dropout = pt.nn.Dropout(config.ff_dropout_ratio)


    def forward(self, x):
        x = self.layer1(x)
        x = pt.nn.functional.gelu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return(x)
    
# Creating the embeddings class, regular + positional ones
class Embeddings(pt.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.regular_embedding = pt.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = pt.nn.Embedding(config.max_pos_emb, config.embedding_dim)

        self.layer_norm = pt.nn.LayerNorm(config.embedding_dim, eps=1e-12)
        self.dropout = pt.nn.Dropout(config.emb_dropout_ratio) #??? book is setting it default

    def forward(self, input_ids):
        # Get the vocab embedings
        reg_emb = self.regular_embedding(input_ids)
        # Get the position embeddings
        input_positions = pt.arange(input_ids.size(1), dtype=pt.long).unsqueeze(0)
        pos_emb = self.positional_embedding(input_positions)
        # Combine embeddings
        embs = reg_emb + pos_emb
        # Layer normalization, why we are not doing very fancy stuff ???
        embs = self.layer_norm(embs)
        # Dropout
        embs + self.dropout(embs)
        return(embs)

# Create a class that holds the config vals
# It initializes them to default vals if not provided


#%%
if __name__ == "__main__":
    from transformers import AutoTokenizer

    #Print config params
    config = Config()
    for k, v in config.__dict__.items():
        print(k, v)
    

    ### Testing the attention heads implementation

    # Load config and tokenizer from model
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Process the input from text to input embeddings
    sentence = "time flies like an arrow"
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)

    ### Testing embedings
    print("\nTesting embeddings")
    print(inputs["input_ids"].size())
    embeddings = Embeddings(config)
    embs = embeddings(inputs["input_ids"])
    print(embs.size())

    ### Testing the attention heads implementation
    print("\nTesting attention head")
    print(embs.size())
    print(config.embedding_dim // config.num_self_attn_heads)
    attention_head = SelfAttentionHead(config.embedding_dim, config.embedding_dim // config.num_self_attn_heads)
    out = attention_head(embs)
    print(out.size())

    ### Testing the multihead attention
    print("\nTesting multihead attention")
    multihead = MultiHeadSelfAttention(config)
    out, k, v = multihead(embs, return_kv=True)
    print(out.size())
    print(k.size())
    print(v.size())

    ### Testing the feed forward
    print("\nTesting feed forward")
    ff = FeedForward(config)
    out = ff(out)
    print(out.size())

    




# %%
