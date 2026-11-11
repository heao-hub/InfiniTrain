"""Run with the model package's PyTorch environment; reads its source only."""
import argparse
import ast
import math
from pathlib import Path

import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="path to modeling_fm9g.py")
args = parser.parse_args()
source = Path(args.model_file).read_text(encoding="utf-8")
tree = ast.parse(source)
names = {"FM9GRotaryEmbedding", "FM9GLongRoPE", "rotate_half", "apply_rotary_pos_emb"}
nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
assert {node.name for node in nodes} == names
scope = {"torch": torch, "nn": nn, "math": math}
exec(compile(ast.Module(body=nodes, type_ignores=[]), args.model_file, "exec"), scope)

rope = scope["FM9GLongRoPE"](4, max_position_embeddings=8, base=100.0,
    short_factor=[1.0, 2.0], long_factor=[3.0, 4.0], original_max_position_embeddings=8)
positions = torch.tensor([[0, 1], [1, 0]])
q = torch.arange(1, 33, dtype=torch.float32).view(2, 2, 2, 4).requires_grad_()
k = torch.arange(1, 17, dtype=torch.float32).view(2, 2, 1, 4).requires_grad_()
cos, sin = rope(k, seq_len=2)
qo, ko = scope["apply_rotary_pos_emb"](q.transpose(1, 2), k.transpose(1, 2), cos, sin, positions)
qo, ko = qo.transpose(1, 2), ko.transpose(1, 2)
qw = (torch.arange(q.numel()).reshape(q.shape) % 7 + 1).float()
kw = (torch.arange(k.numel()).reshape(k.shape) % 7 + 1).float()
((qo * qw).sum() + (ko * kw).sum()).backward()
def analytic(values, weights):
    out, grad = torch.empty_like(values), torch.empty_like(values)
    for b in range(values.shape[0]):
        for t in range(values.shape[1]):
            for h in range(values.shape[2]):
                for j in range(2):
                    angle = float(positions[b, t]) / ([1.0, 2.0][j] * 100.0 ** (j / 2))
                    c, s = math.cos(angle), math.sin(angle)
                    a, z = values[b, t, h, j], values[b, t, h, j + 2]
                    wa, wz = weights[b, t, h, j], weights[b, t, h, j + 2]
                    out[b, t, h, j], out[b, t, h, j + 2] = a * c - z * s, a * s + z * c
                    grad[b, t, h, j], grad[b, t, h, j + 2] = wa * c + wz * s, -wa * s + wz * c
    return out, grad

for actual_out, actual_grad, values, weights in [(qo, q.grad, q.detach(), qw), (ko, k.grad, k.detach(), kw)]:
    expected_out, expected_grad = analytic(values, weights)
    torch.testing.assert_close(actual_out, expected_out, atol=1e-5, rtol=0)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=0)
for name, tensor in [("q_output", qo), ("k_output", ko), ("q_gradient", q.grad), ("k_gradient", k.grad)]:
    print(name, tensor.detach().reshape(-1).tolist())

rope_bf16 = scope["FM9GLongRoPE"](2, max_position_embeddings=2, base=100.0,
    short_factor=[1.0], long_factor=[1.0], original_max_position_embeddings=2)
x = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16).view(1, 1, 2, 2).requires_grad_()
cos, sin = rope_bf16(x, seq_len=2)
q_out, _ = scope["apply_rotary_pos_emb"](x, x.detach(), cos, sin, torch.tensor([[0, 1]]))
q_out.backward(torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16).view_as(q_out))
assert q_out.detach().reshape(-1).float().tolist() == [1.0, 2.0, -1.7421875, 4.6875]
assert x.grad.reshape(-1).float().tolist() == [1.0, 2.0, 4.96875, -0.36328125]
print("bf16_output", q_out.detach().reshape(-1).float().tolist())
print("bf16_gradient", x.grad.reshape(-1).float().tolist())
