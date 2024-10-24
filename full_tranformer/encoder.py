#%%
from config import Config
from common_components import MultiHeadSelfAttention, FeedForward, Embeddings

import torch as pt


# Creating a Transformer Encoder Layer
class TransformerEncoderLayer(pt.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.multi_head_attn = MultiHeadSelfAttention(config)
        self.ff = FeedForward(config)

        # We need to instantiate two different layer norms because we have learnable parameters, since by default elementwise_affine=True
        self.layer_norm1 = pt.nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = pt.nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        y = self.layer_norm1(x)
        y, layer_ks, layer_vs = self.multi_head_attn(y, return_kv=True)
        x = y + x

        y = self.layer_norm2(x)
        y = self.ff(y)
        x = y + x 

        return(x, layer_ks, layer_vs)
    

# The Full Encoder!
class TransformerEncoder(pt.nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.embedding_layer = Embeddings(config)

        self.encoder_layers = pt.nn.ModuleList(
            [
                TransformerEncoderLayer(config) 
                for _ in range(config.num_trans_layers)
            ]
        )
    
    def forward(self, input_ids):

        x = self.embedding_layer(input_ids)

        k_cache = []
        v_cache = []
        for layer in self.encoder_layers:
            x, layer_ks, layer_vs = layer(x)
            k_cache.append(layer_ks.unsqueeze(2)) #(batch, seq_len, 1, num_heads, head_dim)
            v_cache.append(layer_vs.unsqueeze(2)) #(batch, seq_len, 1, num_heads, head_dim)
        
        k_cache = pt.cat(k_cache, dim=2) #(batch, seq_len, layer, num_heads, head_dim)
        v_cache = pt.cat(v_cache, dim=2) #(batch, seq_len, layer, num_heads, head_dim)

        return(x, k_cache, v_cache)


#%%
if __name__ == "__main__":
    from transformers import AutoTokenizer

    #Print config params
    config = Config()
    for k, v in config.__dict__.items():
        print(k, v)
    

    # Load config and tokenizer from model
    model_ckpt = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Process the input from text to input embeddings
    sentence = "time flies like an arrow"
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)

    # Prepare embeddings
    embeddings = Embeddings(config)
    embs = embeddings(inputs["input_ids"])

    ### Testing Encoder Layer
    print("\nTesting Encoder Layer")
    print(embs.size())
    encoder_layer = TransformerEncoderLayer(config)
    out, k, v = encoder_layer(embs)
    print(out.size())
    print(k.size())
    print(v.size())

    ### Testing Encoder
    print("\nTesting Encoder")
    print(inputs["input_ids"].size())
    encoder = TransformerEncoder(config)
    out, k_cache, v_cache = encoder(inputs["input_ids"])
    print(out.size())
    print(k_cache.size())
    print(v_cache.size())

# %%
