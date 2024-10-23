#%%
import torch
import torch.nn.functional as F
import math

# WARRNING: Maybe some namings differ from the book, I tried to write it myself toe classes and then compared the results


def scaled_dot_product_attention(
        query,
        key,
        value
        ) -> torch.Tensor : 

    dim_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)
    att_weights = F.softmax(scores, dim=-1)
    return torch.matmul(att_weights, value)



class AttentionHead(torch.nn.Module):

    def __init__(self, embedding_dim, head_dim):
        super().__init__()

        self.query = torch.nn.Linear(embedding_dim, head_dim)
        self.key = torch.nn.Linear(embedding_dim, head_dim)
        self.value = torch.nn.Linear(embedding_dim, head_dim)

    def forward(self, input_embedding):

        updated_embedding = scaled_dot_product_attention(
            self.query(input_embedding),
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

# 1d convolution / position-wise feed forward 2 layer NN
class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim, latent_dim, dropout_ratio):
        super().__init__()
        self.layer1 = torch.nn.Linear(embedding_dim, latent_dim)
        self.layer2 = torch.nn.Linear(latent_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_ratio)


    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return(x)
    

# Creating a Transformer Encoder Layer
class TransformerEncoderLayer(torch.nn.Module):

    def __init__(self, embedding_dim, attn_n_heads, ff_latent_dim, ff_dropout_ratio):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(embedding_dim, attn_n_heads)
        self.ff = FeedForward(embedding_dim, ff_latent_dim, ff_dropout_ratio)

        # We need to instantiate two different layer norms because we have learnable parameters, since by default elementwise_affine=True
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        y = self.layer_norm1(x)
        y = self.multi_head_attn(y)
        x = y + x

        y = self.layer_norm2(x)
        y = self.ff(y)
        x = y + x 

        return(x)
    
# Creating the embeddings class, regular + positional ones
class Embeddings(torch.nn.Module):

    def __init__(self, vocab_size, max_pos_emb, embedding_dim, emb_dropout_ratio):
        super().__init__()

        self.regular_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = torch.nn.Embedding(max_pos_emb, embedding_dim)

        self.layer_norm = torch.nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(emb_dropout_ratio) #??? book is setting it default

    def forward(self, input_ids):
        # Get the vocab embedings
        reg_emb = self.regular_embedding(input_ids)
        # Get the position embeddings
        input_positions = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        pos_emb = self.positional_embedding(input_positions)
        # Combine embeddings
        embs = reg_emb + pos_emb
        # Layer normalization, why we are not doing very fancy stuff ???
        embs = self.layer_norm(embs)
        # Dropout
        embs + self.dropout(embs)
        return(embs)

# The Full Encoder!
class TransformerEncoder(torch.nn.Module):

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

        self.encoder_layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim, 
                    attn_n_heads, 
                    ff_latent_dim, 
                    ff_dropout_ratio
                ) 
                for _ in range(num_trans_enc_layers)
            ]
        )
    

    def forward(self, input_ids):

        x = self.embedding_layer(input_ids)
        for layer in self.encoder_layers:
            x = layer(x)
        return(x)

#%%
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoConfig

    ### Testing the attention heads implementation

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
    print("Testing the attention heads implementation")
    print(attn_output.size())

    ### Testing the FF layer

    # Instantiating the net
    feed_forward = FeedForward(
        config.hidden_size,
        config.intermediate_size,
        config.hidden_dropout_prob
    )

    ff_outputs = feed_forward(attn_output)
    print("Testing the FF layer")
    print(ff_outputs.size())

    ### Testing TransformerencoderLayer
    encoder_layer = TransformerEncoderLayer(
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.hidden_dropout_prob
    )

    el_outputs = encoder_layer(inputs_embeds)
    print("Testing TransformerencoderLayer")
    print(el_outputs.shape)

    ### Testing embedding layer
    embedding_layer = Embeddings(
        config.vocab_size,
        config.max_position_embeddings,
        config.hidden_size,
        config.hidden_dropout_prob
    )
    emb_outputs = embedding_layer(inputs.input_ids)
    print("Testing Embedding layer")
    print(emb_outputs.shape)

    ### Testing the full encoder
    transformer_encoder = TransformerEncoder(
        config.vocab_size, 
        config.max_position_embeddings,
        config.hidden_size,
        config.hidden_dropout_prob, # I am setting the embedding dropout to config.hidden_dropout_prob, the book leaves the default, which is 0.5 and I think is too much
        config.num_attention_heads,
        config.intermediate_size,
        config.hidden_dropout_prob,
        config.num_hidden_layers
    )

    trans_output = transformer_encoder(inputs.input_ids)
    print("Testing TransformerEncoder!!!!")
    print(trans_output.shape)
    print(trans_output)
# %%
