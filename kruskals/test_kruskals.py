import time
import torch
from kruskals import *


batch_size = 12800
n = 31
num_steps = 1
num_edges = int(n * (n - 1) / 2)
weights = torch.randn(batch_size, num_edges).to("cuda")

# Test pytorch (batched, gpu).
t = 0
vertices = torch.triu_indices(n-1, n, offset=1).to("cuda")
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.cuda().unsqueeze(-1), tiled_vertices], axis=-1)
for _ in range(num_steps):
    start = time.time()
    res_pytorch = kruskals_pytorch_batched(weights_and_edges, n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"Pytorch (batched, gpu): {t}")

# # Test pytorch (batched, cpu).
# t = 0
# vertices = torch.triu_indices(n-1, n, offset=1)
# tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
# weights_and_edges = torch.cat([weights.unsqueeze(-1).to("cpu"), tiled_vertices], axis=-1)
# for _ in range(num_steps):
#     start = time.time()
#     res_pytorch = kruskals_pytorch_batched(weights_and_edges, n)
#     t += time.time() - start
# print(f"Pytorch (batched, cpu): {t}")

# # Test numpy.
# t = 0
# vertices = torch.triu_indices(n-1, n, offset=1)
# tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
# weights_and_edges = torch.cat([weights.to("cpu").unsqueeze(-1), tiled_vertices], axis=-1)
# weights_and_edges_np = onp.array(weights_and_edges)
# for _ in range(num_steps):
#     start = time.time()
#     res_numpy = kruskals_numpy(weights_and_edges_np, n)
#     t += time.time() - start
# print(f"Numpy: {t}")

# Test cpp (pytorch, cpu).
vertices = torch.triu_indices(n-1, n, offset=1)
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.to("cpu").unsqueeze(-1), tiled_vertices], axis=-1)
t = 0
for _ in range(num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch(weights_and_edges, n)
    t += time.time() - start
print(f"C++ (pytorch, cpu): {t}")

# Test cpp (pytorch, gpu).
vertices = torch.triu_indices(n-1, n, offset=1).to("cuda")
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.unsqueeze(-1), tiled_vertices], axis=-1)
t = 0
for _ in range(num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch(weights_and_edges, n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"C++ (pytorch, gpu): {t}")

# Test cpp (pytorch2, cpu).
vertices = torch.triu_indices(n-1, n, offset=1)
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.to("cpu").unsqueeze(-1), tiled_vertices], axis=-1)
t = 0
for _ in range(num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch2(weights_and_edges, n)
    t += time.time() - start
print(f"C++ (pytorch2, cpu): {t}")

# Test cpp (pytorch2, gpu).
vertices = torch.triu_indices(n-1, n, offset=1).to("cuda")
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.unsqueeze(-1), tiled_vertices], axis=-1)
t = 0
for _ in range(num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch2(weights_and_edges, n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"C++ (pytorch2, gpu): {t}")