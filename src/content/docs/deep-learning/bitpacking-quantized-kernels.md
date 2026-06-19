---
title: Bitpacking and Quantized Kernels (GPTQ, AWQ)
description: Explore bitpacking and quantized GPU kernels like GPTQ and AWQ that enable efficient inference for 4-bit and 3-bit Large Language Models.
---

Quantizing Large Language Models to 4-bit or 3-bit weights is essential for running models on consumer GPUs. However, a model's weights cannot be processed directly by GPUs in their quantized state: standard Tensor Cores are designed for 16-bit floating-point (FP16 or BF16) or 8-bit integer (INT8) matrix multiplications.

To bridge this gap, modern inference engines use **Bitpacking** and **Quantized Kernels** (like **GPTQ** and **AWQ**). These techniques pack multiple low-bit weights into standard 32-bit registers and execute customized CUDA/Triton kernels that dequantize weights on-the-fly during matrix multiplication, maximizing memory bandwidth.

---

## The Concept of Bitpacking

In a 4-bit quantized model, each weight requires only 4 bits of memory. However, computer memory is addressed in bytes (8 bits) or words (32/64 bits). 

**Bitpacking** is the process of packing multiple low-bit values into a single larger integer data type. For example, we can pack eight 4-bit weight integers into a single 32-bit integer (`uint32`):

```
32-bit Register:
[ w0 (4b) | w1 (4b) | w2 (4b) | w3 (4b) | w4 (4b) | w5 (4b) | w6 (4b) | w7 (4b) ]
```

During inference, instead of loading eight separate 16-bit floats ($8 \times 16 = 128\text{ bits}$), the GPU loads a single 32-bit integer from VRAM—**a 4x savings in memory bandwidth**.

---

## On-the-Fly Dequantization Kernels

Since the actual computation must happen in FP16, a specialized GPU kernel dequantizes the weights in register memory right before the matrix multiplication.

For a weight parameter $W_q$ packed within a 32-bit word:
1. **Unpacking:** The kernel uses bitwise shift (`>>`) and mask (`&`) operations to isolate the 4-bit integer values in GPU registers.
2. **Scaling and Bias:** The unpacked integer value is converted to a floating-point value and scaled using group-level scale ($S$) and zero-point ($Z$) parameters:
   
   $$W_{\text{FP16}} = (W_q - Z) \times S$$

3. **Computation:** The dequantized weight is multiplied by the FP16 activation input: $y = W_{\text{FP16}} \times x_{\text{FP16}}$.

Because this dequantization happens entirely inside the GPU's fast register memory, it avoids slow VRAM round-trips. Since LLM inference is highly memory-bandwidth bound rather than compute bound, the extra ALU cycles spent on dequantization are fully hidden by the memory savings.

---

## GPTQ vs. AWQ: Two Quantization Paradigms

While both GPTQ and AWQ use bitpacking and custom kernels, they differ in how they compute the quantized weights during calibration.

### 1. GPTQ (Generalized Post-Training Quantization)
GPTQ is an second-order optimization method based on the **Approximate Inverse Hessian** matrix. It quantizes weights column-by-column and dynamically adjusts the remaining unquantized weights in the matrix to compensate for the quantization error introduced in previous columns:

$$\Delta w_i = -\frac{w_i - \text{round}(w_i)}{[H^{-1}]_{i,i}} \cdot H^{-1}_{:,i}$$

GPTQ produces highly accurate 4-bit and 3-bit weights but requires a calibration step that takes 1 to 2 hours for large models.

### 2. AWQ (Activation-aware Weight Quantization)
AWQ is based on the observation that **not all weights in an LLM are equally important**. Only 1% of the weights (corresponding to channels with large activation magnitudes) dominate the model's performance.

Instead of adjusting all weights, AWQ:
- Identifies the top-1% salient weights.
- Scales up the activations of these salient channels by a factor $s > 1$, and scales down the corresponding weights by $1/s$. This reduces the relative quantization error on these critical channels.
- Performs standard round-to-nearest quantization on the rest.

AWQ is fast (takes minutes to run) and generalizes well across diverse datasets.

---

## Comparison of Quantized Implementations

| Feature | AutoGPTQ (GPTQ) | AutoAWQ (AWQ) | vLLM (Marlin/GPTQ) |
|---|---|---|---|
| **Calibration Method** | Second-order Hessian error compensation | Activation-aware scaling | Mixed (Marlin is an optimized kernel format) |
| **Quantization Speed**| Slow (hours) | Fast (minutes) | N/A (runs pre-quantized models) |
| **Kernel Performance**| Moderate | High (optimized GEMM kernels) | Extremely High (Marlin kernels) |
| **Group Size Support**| 128, 64, 32, -1 (none) | 128 (standard) | 128 (highly optimized) |

---

## Python Concept: Simulating Bit-Unpacking in PyTorch

Below is a conceptual Python snippet showing how 4-bit packed weights are unpacked into FP16 values using bitwise operations.

```python
import torch

def unpack_4bit_weights(packed_weights, scales, zero_points, group_size=128):
    """
    Simulates unpacking of 4-bit weights stored in a uint32 tensor.
    packed_weights: [N, K // 8] of type torch.int32 (each element holds eight 4-bit weights)
    """
    N, packed_K = packed_weights.shape
    K = packed_K * 8
    
    # Initialize output FP16 weight matrix
    unpacked_weights = torch.zeros((N, K), dtype=torch.float16, device=packed_weights.device)
    
    # Shift values to extract the eight 4-bit segments:
    # Segment 0: bits 0-3, Segment 1: bits 4-7, ... Segment 7: bits 28-31
    for i in range(8):
        shift = i * 4
        # Shift right and mask out the last 4 bits (0x0F)
        extracted = (packed_weights >> shift) & 0x0F
        
        # Write back to corresponding columns in the output matrix
        unpacked_weights[:, i::8] = extracted.to(torch.float16)
        
    # Apply scales and zero points group-wise
    # (In a real GPU kernel, this is parallelized block-wise)
    for g in range(K // group_size):
        start_col = g * group_size
        end_col = (g + 1) * group_size
        
        g_scale = scales[:, g:g+1]
        g_zero = zero_points[:, g:g+1]
        
        # Dequantize: W_fp16 = (W_q - Z) * S
        unpacked_weights[:, start_col:end_col] = (
            (unpacked_weights[:, start_col:end_col] - g_zero) * g_scale
        )
        
    return unpacked_weights
```
