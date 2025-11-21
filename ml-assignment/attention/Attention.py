import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Implements Scaled Dot-Product Attention using only NumPy.

    Args:
        Q (numpy.ndarray): Query matrix with shape (..., seq_len_q, d_k)
        K (numpy.ndarray): Key matrix with shape   (..., seq_len_k, d_k)
        V (numpy.ndarray): Value matrix with shape (..., seq_len_v, d_v)
        mask (numpy.ndarray, optional):
            Mask broadcastable to (..., seq_len_q, seq_len_k).
            Used for padding or causal attention.

    Returns:
        output (numpy.ndarray): Attention applied to V
        attention_weights (numpy.ndarray): Softmax attention weights
    """

    # ----------------------------------------------------
    # 1. Dot Product between Q and K^T
    # ----------------------------------------------------
    # Computes Q * K^T -> attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # ----------------------------------------------------
    # 2. Scale by sqrt(d_k)
    # ----------------------------------------------------
    d_k = Q.shape[-1]
    scores = scores / np.sqrt(d_k)

    # ----------------------------------------------------
    # 3. Apply mask (if provided)
    # ----------------------------------------------------
    if mask is not None:
        # Mask = 0 → add -inf (or a large negative number)
        scores = np.where(mask == 0, -1e9, scores)

    # ----------------------------------------------------
    # 4. Apply softmax over last axis to get attention weights
    # ----------------------------------------------------
    # Subtract max for numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # ----------------------------------------------------
    # 5. Multiply attention weights with V
    # ----------------------------------------------------
    output = np.matmul(attention_weights, V)

    return output, attention_weights
