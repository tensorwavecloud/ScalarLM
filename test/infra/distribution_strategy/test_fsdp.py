import torch
import torch.nn as nn
import numpy as np
import logging
import random
from gpu_aware_mpi import get_rank, finalize_mpi

from infra.cray_infra.training.distribution_strategy.fsdp.fsdp import SimpleFSDP

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a fixed seed
seed_all(42)

def test_sequential_model(rank, device):
    class SequentialModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
            super(SequentialModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.model(x)
    
    fsdp_model = SimpleFSDP(SequentialModel()).to(device)
    
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        input_data = torch.randn(32, 10, device=device)
        output = fsdp_model(input_data)
        
        target = torch.randn(32, 5, device=device)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
    
    assert isinstance(fsdp_model, SimpleFSDP)
    assert loss.item() > 0

if __name__ == "__main__":
    rank = get_rank()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_sequential_model(rank, device)
    finalize_mpi()

# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 2 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 4 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 8 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
