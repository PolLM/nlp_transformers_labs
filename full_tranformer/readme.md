# Transformer Encoder-Decoder Implementation

This is my first attempt at implementing a full Encoder-Decoder Transformer architecture from the ground up, despite having used extensively large language models (LLMs). The goal is to adapt the `TransformerEncoder` from Chapter 3 ("Transformer Anatomy") of the book and create a corresponding `Decoder` to build an `EncoderDecoder Transformer`.

Warning: My approach is to implement it myself, learn from mistakes, and then review how it is properly done to maximize learning. Do not take this as a reference for best practices or optimal implementation.


## Current Implementation

In the `full_transformer/common_components.py` file, I have defined the common components that will be used in both the encoder and decoder: 
- `ScaledDotProductAttention`
- `SelfAttentionHead`
- `MultiHeadSelfAttention`
- `FeedForward`
- `Embeddings` (with positional and token embeddings, Important to notice that I am assuming encoder and decoder have the same vocabulary size and embedding dimensions, not ideal for real-world applications)

In the `full_transformer/encoder.py` file, I have implemented the `TransformerEncoder` class, which is a stack of `TransformerEncoderLayer` instances. I have adapted this class from Chapter 3 of the book to store and output the key and value tensors from each layer, which will be used by the decoder.

In the `full_transformer/decoder.py` file, I have implemented the `TransformerDecoder` class, which is a stack of `TransformerDecoderLayer` instances.

## Next Steps

- Implement the `TranformerEncoderDecoder` class that will combine the encoder and decoder.
- Define the learning task and dataset.
- Configure the Optimizer and Loss function.
- Train the model and evaluate its performance.





---
