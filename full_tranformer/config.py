class Config:
        def __init__(self, **kwargs):
            self.num_self_attn_heads = kwargs.get("num_self_attn_heads", 12)
            self.embedding_dim = kwargs.get("embedding_dim", 768)
            self.ff_latent_dim = kwargs.get("latent_dim", 3072)
            self.ff_dropout_ratio = kwargs.get("ff_dropout_ratio", 0.1)
            self.emb_dropout_ratio = kwargs.get("emb_dropout_ratio", 0.1)
            self.vocab_size = kwargs.get("vocab_size", 30522)
            self.max_pos_emb = kwargs.get("max_pos_emb", 512)
            self.num_trans_layers = kwargs.get("num_trans_layers", 6)