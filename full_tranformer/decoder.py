#%%
# Here I will adapt the encoder code to make the decoder
# To implement the decoder I have adapted the scaled dot product as indicated in the book 
# We also need to have access to the encoder's K, V values used by the scond attention step in the decoder. 
# We have to create a KV cache for the Encoder. It will be a dict of the form kv_cache[enc_layer][keys/vals][enc_head]

import torch as pt
from common_components import scaled_dot_product_attention, Embeddings, FeedForward, MultiHeadSelfAttention
from config import Config
from encoder import TransformerEncoder

class EncoderDecoderAttentionHead(pt.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        # the query is computed in the decoder and will have the form (batch, dec_seq_len, embedding_dim)
        # keys and values come from the encoder and have the form (batch, enc_seq_len, embedding_dim) they are already computed

        self.query = pt.nn.Linear(embedding_dim, head_dim)


    def forward(self, input_embedding, enc_key, enc_value):
        # Here query has shape (batch, dec_seq_len, head_dim)
        # enc_key and enc_value have shape (batch, enc_seq_len, head_dim)

        updated_embedding = scaled_dot_product_attention(
            self.query(input_embedding),
            enc_key,
            enc_value
        )

        # updated_embedding should have dim [batch, seq_len, head_dim]
        return(updated_embedding)
    

class MultiHeadEncoderDecoderAttention(pt.nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.embedding_dim % config.num_self_attn_heads != 0:
            raise ValueError(f"embedding dimension {config.embedding_dim} is not divisible by num_heads {config.num_self_attn_heads}")
        
        head_dim = config.embedding_dim // config.num_self_attn_heads

        # Had to look what moduleList does, I used a list at first
        # Is this exploiting the parallellization opportunity of multiple independent Heads ???
        self.heads = pt.nn.ModuleList(
            [EncoderDecoderAttentionHead(config.embedding_dim, head_dim) for _ in range(config.num_self_attn_heads)]
        )
        # I do not get why we are applying the linear, it is not changing the simd of the array ???
        self.out_linear = pt.nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, input_embedding, layer_enc_k, layer_enc_v):

        # layer_enc_k and layer_enc_v have shape (batch, enc_seq_len, num_heads, head_dim)

        # here I assume that the enc_keys and enc_values are a list of length n_heads
        # Get all the heads processed
        out_heads = [head(input_embedding, layer_enc_k[:,:,i,:], layer_enc_v[:,:,i,:]) for i, head in enumerate(self.heads)]
        # Concatenate the out heads
        x = pt.cat(out_heads, dim=-1)
        # Linear projection 
        x = self.out_linear(x)
        return(x)
    

# It is not clearly specified the decoder exact transformations so I have decided to apply this decoder architecture:
# skipcon1_start x (layer_norm x multihead_self_attn) x skipcon1_end x skipcon2_start x (layer_norm x multihead_encdec_attn) x skipcon2_end x skipcon3_start x (layer_norm x PW_FF_NN) x skipcon3_end

class TransformerDecoderLayer(pt.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head_self_attn = MultiHeadSelfAttention(config)
        self.multi_head_encdec_attn = MultiHeadEncoderDecoderAttention(config)

        self.ff = FeedForward(config)

        # We need to instantiate two different layer norms because we have learnable parameters, since by default elementwise_affine=True
        self.layer_norm1 = pt.nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = pt.nn.LayerNorm(config.embedding_dim)
        self.layer_norm3 = pt.nn.LayerNorm(config.embedding_dim)

    def forward(self, x, layer_enc_k, layer_enc_v):
        y = self.layer_norm1(x)
        y = self.multi_head_self_attn(y)
        x = y + x

        y = self.layer_norm2(x)
        y = self.multi_head_encdec_attn(y, layer_enc_k, layer_enc_v)
        x = y + x

        y = self.layer_norm3(x)
        y = self.ff(y)
        x = y + x 

        return(x)
    

class TransformerDecoder(pt.nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.embedding_layer = Embeddings(config)

        self.decoder_layers = pt.nn.ModuleList(
            [
                TransformerDecoderLayer(config) for _ in range(config.num_trans_layers)
            ]
        )
    

    def forward(self, input_ids, k_cache, v_cache):
        # k_cache, v_cache have shape (batch, enc_seq_len, num_layers, num_heads, head_dim)

        x = self.embedding_layer(input_ids)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, k_cache[:,:,i,:,:], v_cache[:,:,i,:,:])
        return(x)
    
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
        enc_sentence = "time flies like an arrow"
        enc_inputs = tokenizer(enc_sentence, return_tensors="pt", add_special_tokens=False)

        dec_sentence = "el temps passa "
        dec_inputs = tokenizer(dec_sentence, return_tensors="pt", add_special_tokens=False)

        # Forward encoder
        print("\nTesting Encoder")
        print(enc_inputs["input_ids"].size())
        encoder = TransformerEncoder(config)
        _, k_cache, v_cache = encoder(enc_inputs["input_ids"])
        print(out.size())
        print(k_cache.size())
        print(v_cache.size())


        ### Forward decoder sentence through the embeddings
        print("\nProcessing embeddings decoder input to test dec modules")
        print(dec_inputs["input_ids"].size())
        embeddings = Embeddings(config)
        dec_embs = embeddings(dec_inputs["input_ids"])
        print(dec_embs.size())

        ### Testing the EncoderDecoderAttentionHead
        print("\nTesting EncoderDecoderAttentionHead")
        print(dec_embs.size())
        enc_dec_attention_head = EncoderDecoderAttentionHead(config.embedding_dim, config.embedding_dim // config.num_self_attn_heads)
        out = enc_dec_attention_head(dec_embs, k_cache[:,:,0,0,:], v_cache[:,:,0,0,:]) # Fixing the first layer and first head
        print(out.size())

        ### Testing the MultiHeadEncoderDecoderAttention
        print("\nTesting MultiHeadEncoderDecoderAttention")
        print(dec_embs.size())
        multihead_encdec = MultiHeadEncoderDecoderAttention(config)
        out = multihead_encdec(dec_embs, k_cache[:,:,0,:,:], v_cache[:,:,0,:,:])
        print(out.size())

        ### Testing the DecoderLayer
        print("\nTesting DecoderLayer")
        print(dec_embs.size())
        decoder_layer = TransformerDecoderLayer(config)
        out = decoder_layer(dec_embs, k_cache[:,:,0,:,:], v_cache[:,:,0,:,:])
        print(out.size())

        ### Testing the Decoder !!!!!!
        print("\nTesting Decoder")
        print(dec_inputs["input_ids"].size())
        decoder = TransformerDecoder(config)
        out = decoder(dec_inputs["input_ids"], k_cache, v_cache)
        print(out.size())

        # print model info 
        print(decoder)

# %%
