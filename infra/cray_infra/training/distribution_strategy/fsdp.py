from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn

class FSDPLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()

    def all_gather(self, tensor):
        if tensor is None:
            return None
        tensor_numpy = tensor.detach().numpy()
        gathered = np.zeros([self.world_size] + list(tensor_numpy.shape), dtype=tensor_numpy.dtype)
        self.comm.Allgather(tensor_numpy, gathered)
        return torch.from_numpy(gathered).to(tensor.device)

    def forward(self, x):
        if hasattr(self.layer, 'weight'):
            full_weight = self.all_gather(self.layer.weight)
            self.layer.weight.data = full_weight[self.rank]
        if hasattr(self.layer, 'bias'):
            full_bias = self.all_gather(self.layer.bias)
            self.layer.bias.data = full_bias[self.rank]
        return self.layer(x)

    def synchronize_gradients(self):
        def reduce_scatter(param):
            if param.grad is not None:
                grad = param.grad.data
                local_size = grad.numel() // self.world_size
                start = self.rank * local_size
                end = start + local_size if self.rank < self.world_size - 1 else grad.numel()
                
                local_grad = torch.zeros(end - start, dtype=grad.dtype, device=grad.device)
                
                self.comm.Reduce_scatter(grad.view(-1).numpy(), local_grad.numpy(), op=MPI.SUM)
                
                grad.view(-1)[start:end] = local_grad
                grad /= self.world_size
                param.grad.data = grad

        if hasattr(self.layer, 'weight'):
            reduce_scatter(self.layer.weight)
        if hasattr(self.layer, 'bias'):
            reduce_scatter(self.layer.bias)

class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(self._get_fsdp_layers(model))

    def _get_fsdp_layers(self, model):
        return [FSDPLayer(module) for module in model.modules() if self._is_leaf_module(module)]

    @staticmethod
    def _is_leaf_module(module):
        return not list(module.children())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss):
        loss.backward()
        for layer in reversed(self.layers):
            layer.synchronize_gradients()

