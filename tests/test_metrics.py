import torch
from torch import nn

from llm_compression_benchmark.utils import effective_nonzero_bytes, model_sparsity, tensor_storage_bytes


def test_sparsity_and_effective_bytes():
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]])
    assert model_sparsity(model) == 0.5
    assert effective_nonzero_bytes(model) == 4 * model.weight.element_size()
    assert tensor_storage_bytes(model) == model.weight.numel() * model.weight.element_size()
