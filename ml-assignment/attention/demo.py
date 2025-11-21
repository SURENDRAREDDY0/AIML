import numpy as np
from attention import scaled_dot_product_attention

# Sample data: batch=1, seq_len=3, depth=4
Q = np.array([[
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
]], dtype=float)

K = np.array([[
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0]
]], dtype=float)

V = np.array([[
    [10, 0],
    [0, 10],
    [5, 5]
]], dtype=float)

# Optional mask (uncomment to test)
# mask = np.array([[[1, 0, 0],
#                   [1, 1, 0],
#                   [1, 1, 1]]])
mask = None

output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

print("\nAttention Weights:\n", attn_weights)
print("\nOutput:\n", output)
