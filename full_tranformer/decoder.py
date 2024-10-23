# Here I will adapt the encoder code to make the decoder
# To implement the decoder I have adapted the scaled dot product as indicated in the book 
# We also need to have access to the encoder's K, V values used by the scond attention step in the decoder. 
# We have to create a KV cache for the Encoder. It will be a dict of the form kv_cache[enc_layer][keys/vals][enc_head]

import torch
from transformer_components import scaled_dot_product_attention, Embeddings, FeedForward


#Let's build the decoder first by doing:
# 1. Update the AttentionHead class to SelfAttentionHead with the mask
# 2. Update the MultiHeadAttention class to MultiHeadSelfAttention with the mask
# 3. Create a new class called EncoderDecoderAttentionHead 
# 4. Create a new class called MultiHeadEncoderDecoderAttention 
# 5. Create a class called TransformerDecoderLayer
# 6. Group all of it under TransformerDecoder class

# Then to finish we will:
# 7. Adapt the encoder to output the KV cache
# 8. Create a full fledged TransformerEncoderDecoder !!!!

class SelfAttentionHead(torch.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.quey = torch.nn.Linear(embedding_dim, head_dim)
        self.key = torch.nn.Linear(embedding_dim, head_dim)
        self.value = torch.nn.Linear(embedding_dim, head_dim)

    def forward(self, input_embedding, mask):

        updated_embedding = scaled_dot_product_attention(
            self.quey(input_embedding),
            self.key(input_embedding),
            self.value(input_embedding),
            mask
        )

        # updated_embedding should have dim [batch, seq_len, head_dim]
        return(updated_embedding)

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding dimension {embedding_dim} is not divisible by n_heads {n_heads}")
        
        head_dim = embedding_dim // n_heads

        # Had to look what moduleList does, I used a list at first
        # Is this exploiting the parallellization opportunity of multiple independent Heads ???
        self.heads = torch.nn.ModuleList(
            [SelfAttentionHead(embedding_dim, head_dim) for _ in range(n_heads)]
        )
        # I do not get why we are applying the linear, it is not changing the simd of the array ???
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embedding):

        seq_len = input_embedding.shape(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        # Get all the heads processed
        out_heads = [head(input_embedding, mask) for head in self.heads]
        # Concatenate the out heads
        x = torch.cat(out_heads, dim=-1)
        # Linear projection 
        x = self.out_linear(x)
        return(x)
    

class EncoderDecoderAttentionHead(torch.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        # the query is computed in the decoder and will have the form (batch, dec_seq_len, embedding_dim)
        # keys and values come from the encoder and have the form (batch, enc_seq_len, embedding_dim) they are already computed

        self.query = torch.nn.Linear(embedding_dim, head_dim)


    def forward(self, input_embedding, enc_key, enc_value):

        updated_embedding = scaled_dot_product_attention(
            self.query(input_embedding),
            enc_key(input_embedding),
            enc_value(input_embedding)
        )

        # updated_embedding should have dim [batch, seq_len, head_dim]
        return(updated_embedding)
    

class MultiHeadEncoderDecoderAttention(torch.nn.Module):

    def __init__(self, embedding_dim, n_heads):
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(f"embedding dimension {embedding_dim} is not divisible by n_heads {n_heads}")
        
        head_dim = embedding_dim // n_heads

        # Had to look what moduleList does, I used a list at first
        # Is this exploiting the parallellization opportunity of multiple independent Heads ???
        self.heads = torch.nn.ModuleList(
            [EncoderDecoderAttentionHead(embedding_dim, head_dim) for _ in range(n_heads)]
        )
        # I do not get why we are applying the linear, it is not changing the simd of the array ???
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embedding, layer_kv_cache):

        enc_keys = layer_kv_cache["keys"]
        enc_values = layer_kv_cache["vals"]

        # here I assume that the enc_keys and enc_values are a list of length n_heads
        # Get all the heads processed
        out_heads = [head(input_embedding, enc_keys[i], enc_values[i]) for i, head in enumerate(self.heads)]
        # Concatenate the out heads
        x = torch.cat(out_heads, dim=-1)
        # Linear projection 
        x = self.out_linear(x)
        return(x)
    

# It is not clearly specified the decoder exact transformations so I have decided to apply this decoder architecture:
# skipcon1_start x (layer_norm x multihead_self_attn) x skipcon1_end x skipcon2_start x (layer_norm x multihead_encdec_attn) x skipcon2_end x skipcon3_start x (layer_norm x PW_FF_NN) x skipcon3_end

class TransformerDecoderLayer
    def __init__(self, embedding_dim, attn_n_heads, ff_latent_dim, ff_dropout_ratio):
        super().__init__()
        self.multi_head_self_attn = MultiHeadSelfAttention(embedding_dim, attn_n_heads)
        self.multi_head_encdec_attn = MultiHeadEncoderDecoderAttention(embedding_dim, attn_n_heads)

        self.ff = FeedForward(embedding_dim, ff_latent_dim, ff_dropout_ratio)

        # We need to instantiate two different layer norms because we have learnable parameters, since by default elementwise_affine=True
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm3 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x, layer_kv_cache):
        y = self.layer_norm1(x)
        y = self.multi_head_self_attn(y)
        x = y + x

        y = self.layer_norm2(x)
        y = self.multi_head_encdec_attn(y, layer_kv_cache)
        x = y + x

        y = self.layer_norm3(x)
        y = self.ff(y)
        x = y + x 

        return(x)
    

class TransformerDecoder(torch.nn.Module):

    def __init__(
            self, 
            vocab_size, 
            max_pos_emb, 
            embedding_dim, 
            emb_dropout_ratio, 
            attn_n_heads, 
            ff_latent_dim,
            ff_dropout_ratio,
            num_trans_enc_layers
            ):
        
        super().__init__()
        self.embedding_layer = Embeddings(
            vocab_size, 
            max_pos_emb, 
            embedding_dim,
            emb_dropout_ratio
        )

        self.decoder_layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embedding_dim, 
                    attn_n_heads, 
                    ff_latent_dim, 
                    ff_dropout_ratio
                ) 
                for _ in range(num_trans_enc_layers)
            ]
        )
    

    def forward(self, input_ids, kv_cache):

        x = self.embedding_layer(input_ids)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, kv_cache[i])
        return(x)