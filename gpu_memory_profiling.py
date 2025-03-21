import torch

print(f"GPU Memory Allocated (before): {torch.cuda.memory_allocated() / 1e6:.2f} MB")

# Simulate KV cache creation
K = torch.randn(2048, 512, device="cuda")
V = torch.randn(2048, 512, device="cuda")
cache = {"K": K, "V": V}

print(f"GPU Memory Allocated (after): {torch.cuda.memory_allocated() / 1e6:.2f} MB")