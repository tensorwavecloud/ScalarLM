import torch
import torch.nn as nn
import numpy as np
from mpi4py import MPI

class FSDPLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.backward_handle = self.register_backward_hook(self.synchronize_gradients_hook)
        self.forward_handle = self.module.register_forward_pre_hook(self.forward_pre_hook)

    def all_gather(self, tensor):
        if tensor is None:
            return None
        tensor_numpy = tensor.detach().to(torch.float32).numpy()
        gathered = np.zeros([self.world_size] + list(tensor_numpy.shape), dtype=tensor_numpy.dtype)
        self.comm.Allgather(tensor_numpy, gathered)
        return torch.from_numpy(gathered).to(tensor.dtype).to(tensor.device)

    def forward_pre_hook(self, module, input):
        if hasattr(module, 'weight'):
            full_weight = self.all_gather(module.weight)
            module.weight.data = full_weight[self.rank]
        if hasattr(module, 'bias') and module.bias is not None:
            full_bias = self.all_gather(module.bias)
            module.bias.data = full_bias[self.rank]
        return input

    def reduce_scatter(self, param):
        if param.grad is not None:
            grad = param.grad.data
            local_size = grad.numel() // self.world_size
            start = self.rank * local_size
            end = start + local_size if self.rank < self.world_size - 1 else grad.numel()
            
            local_grad = torch.zeros(end - start, dtype=grad.dtype, device=grad.device)
            
            self.comm.Reduce_scatter(grad.view(-1).numpy(), local_grad.numpy(), op=MPI.SUM)
            
            grad.view(-1)[start:end] = local_grad.to(grad.device)
            grad /= self.world_size
            param.grad.data = grad

    def synchronize_gradients_hook(self, module, grad_input, grad_output):
        if hasattr(module, 'weight'):
            self.reduce_scatter(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            self.reduce_scatter(module.bias)
        return grad_input

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def remove_hooks(self):
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()

class SimpleFSDP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._wrap_layers(model)

    def _wrap_layers(self, module):
        for name, child in module.named_children():
            if list(child.children()):
                self._wrap_layers(child)
            else:
                setattr(module, name, FSDPLayer(child))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def remove_all_hooks(self):
        for module in self.model.modules():
            if isinstance(module, FSDPLayer):
                module.remove_hooks()
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def unwrap_model(self):
        config = self.model.config
        unwrapped_model = type(self.model)(config)
        unwrapped_state_dict = {}

        for name, module in self.model.named_modules():
            if isinstance(module, FSDPLayer):
                if hasattr(module.module, 'weight'):
                    full_weight = module.all_gather(module.module.weight)
                    # Remove 'layer.' from the key name
                    adjusted_name = name.replace('mlp.layer.', 'mlp.')
                    unwrapped_state_dict[f"{adjusted_name}.weight"] = full_weight[0]
                if hasattr(module.module, 'bias') and module.module.bias is not None:
                    full_bias = module.all_gather(module.module.bias)
                    # Remove 'layer.' from the key name
                    adjusted_name = name.replace('mlp.layer.', 'mlp.')
                    unwrapped_state_dict[f"{adjusted_name}.bias"] = full_bias[0]

        # Load the gathered state dict into the new model
        unwrapped_model.load_state_dict(unwrapped_state_dict, strict=False)
        
        # Fix for pad_token_id
        unwrapped_model.config.pad_token_id = unwrapped_model.config.eos_token_id
        if hasattr(unwrapped_model, 'generation_config'):
            unwrapped_model.generation_config.pad_token_id = unwrapped_model.config.eos_token_id

        return unwrapped_model
    