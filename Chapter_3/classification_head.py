#%%
from transformer_components import TransformerEncoder
import torch


class TransformerForSequenceClassification(torch.nn.Module):

    def __init__(
            self, 
            num_classes,
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
        self.encoder = TransformerEncoder(
                        vocab_size, 
                        max_pos_emb, 
                        embedding_dim, 
                        emb_dropout_ratio, 
                        attn_n_heads, 
                        ff_latent_dim,
                        ff_dropout_ratio,
                        num_trans_enc_layers
        )

        self.dropout = torch.nn.Dropout(ff_dropout_ratio)

        self.classifier = torch.nn.Linear(embedding_dim, num_classes)


    def forward(self, input_ids):
        x = self.encoder(input_ids)
        x = self.dropout(x)
        # I do not quite like to use only the hidden state of the first token to perform the classification
        # the Transformer mixes information of the embedding and makes sense to think that the start token contains info of the whole sentence but
        # I will average the embedding and use this to classify
        x = torch.mean(x, dim=1) 
        x = self.classifier(x)
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

    # Use the encoder classifier
    classifier = TransformerForSequenceClassification(
        3,
        config.vocab_size, 
        config.max_position_embeddings,
        config.hidden_size,
        config.hidden_dropout_prob, # I am setting the embedding dropout to config.hidden_dropout_prob, the book leaves the default, which is 0.5 and I think is too much
        config.num_attention_heads,
        config.intermediate_size,
        config.hidden_dropout_prob,
        config.num_hidden_layers
    )

    logits = classifier(inputs.input_ids)
    print("Testing Classifier")
    print(logits.shape)
    print(logits)
# %%
