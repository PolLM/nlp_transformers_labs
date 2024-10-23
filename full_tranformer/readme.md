# Transformer Encoder-Decoder Implementation

This is my first attempt at implementing a full Encoder-Decoder Transformer architecture from the ground up, despite having used extensively large language models (LLMs). The goal is to adapt the `TransformerEncoder` from Chapter 3 ("Transformer Anatomy") of the book and create a corresponding `Decoder` to build an `EncoderDecoder Transformer`.

Warning: My approach is to implement it myself, learn from mistakes, and then review how it is properly done to maximize learning. Do not take this as a reference for best practices or optimal implementation.


### 1. Build the Decoder

In the `full_transformer/decoder.py` file, we will create the decoder by following these steps:

- **Update the `AttentionHead` class**:
  - Modify it to become `SelfAttentionHead` with masking capability.
  
- **Update the `MultiHeadAttention` class**:
  - Modify it to become `MultiHeadSelfAttention` with masking capability.
  
- **Create an `EncoderDecoderAttentionHead` class**:
  - This class will handle attention between encoder outputs (KV cache) and decoder inputs (queries).
  
- **Create a `MultiHeadEncoderDecoderAttention` class**:
  - This class will handle multi-head attention between the encoder's key-value pairs and the decoder's queries.
  
- **Create a `TransformerDecoderLayer` class**:
  - This layer will serve as the building block for the decoder, combining encoder-decoder attention and self-attention mechanisms.
  
- **Group everything into a `TransformerDecoder` class**:
  - This will manage the full decoding process, utilizing the layers and attention mechanisms defined above.

### 2. Adapt the Encoder

In the `full_transformer/encoder.py` file:

- **Adapt the encoder** to output the Key-Value (KV) cache:
  - Modify the encoder to store and output the KV pairs at each layer, which the decoder will later use.

### 3. Build the Full Transformer

In the `full_transformer/transformer.py` file:

- **Create a full-fledged `TransformerEncoderDecoder`**:
  - Combine the `TransformerEncoder` and `TransformerDecoder` into a unified model, passing the KV cache from the encoder to the decoder.

---
