The notebook 2_queries_keys_values.ipynb does not load well on the withub browser (I guess because there are LateX expressions), so I add them here: 

### Queries, Keys and Values

I will ommit the batch dimension, I will assume that the input embeddings are of shape (seq_length, d_e) and the output embeddings are of shape (seq_length, d_emb).

I will do some math to understand it better (also looking at https://arxiv.org/abs/1706.03762):

Notation:

* Values matrix: $V \in \mathbb{R}^{seq\_length \times d_{e}}$
* Single value vector: $v_i \in \mathbb{R}^{d_{e}}$
* Keys matrix: $K \in \mathbb{R}^{seq\_length \times d_{k}}$+
* Single key vector: $k_i \in \mathbb{R}^{d_{k}}$
* Queries matrix: $Q \in \mathbb{R}^{seq\_length \times d_{q}}$
* Single query vector: $q_i \in \mathbb{R}^{d_{q}}$


$$V = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_{seq\_length} \end{bmatrix} K = \begin{bmatrix} k_1 \\ k_2 \\ \vdots \\ k_{seq\_length} \end{bmatrix} Q = \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_{seq\_length} \end{bmatrix}$$

1. **Dot product between queries and keys**:

$$ \frac{Q \cdot K^T}{\sqrt{d_{k}}} = \frac{1}{\sqrt{d_{k}}} \begin{bmatrix} q_1 \cdot k_1 & q_1 \cdot k_2 & \cdots & q_1 \cdot k_{seq\_length} \\ q_2 \cdot k_1 & q_2 \cdot k_2 & \cdots & q_2 \cdot k_{seq\_length} \\ \vdots & \vdots & \ddots & \vdots \\ q_{seq\_length} \cdot k_1 & q_{seq\_length} \cdot k_2 & \cdots & q_{seq\_length} \cdot k_{seq\_length} \end{bmatrix}$$

2. **Softmax**:

$$ W = \text{softmax} \left( \frac{Q \cdot K^T}{\sqrt{d_{k}}} \right) = \begin{bmatrix} w_{1,1} & w_{1,2} & \cdots & w_{1,seq\_length} \\ w_{2,1} & w_{2,2} & \cdots & w_{2,seq\_length} \\ \vdots & \vdots & \ddots & \vdots \\ w_{seq\_length,1} & w_{seq\_length,2} & \cdots & w_{seq\_length,seq\_length} \end{bmatrix} $$

$$where \sum_{j = 1}^{seq\_length} w_{ij} = 1 \quad \forall i  $$

3. **Weighted sum of values**:

$$ X^{'} =  \begin{bmatrix} x^{'}_1 \\ x^{'}_2 \\ \vdots \\ x^{'}_{seq\_length} \end{bmatrix} = W \cdot V = \begin{bmatrix} w_{1,1} & w_{1,2} & \cdots & w_{1,seq\_length} \\ w_{2,1} & w_{2,2} & \cdots & w_{2,seq\_length} \\ \vdots & \vdots & \ddots & \vdots \\ w_{seq\_length,1} & w_{seq\_length,2} & \cdots & w_{seq\_length,seq\_length} \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_{seq\_length} \end{bmatrix} $$

$$so, \quad x^{'}_i = \sum_{j = 1}^{seq\_length} w_{ij} \cdot v_j $$
