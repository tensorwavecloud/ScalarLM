import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import numpy as np
import logging
import random
import math
from mpi4py import MPI

from infra.cray_infra.training.distribution_strategy.fsdp import SimpleFSDP

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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

# Define a simple transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(SimpleTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) 
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_decoder(src, src, tgt_mask=None)
        output = self.decoder(output)
        return output  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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
    
    fsdp_model = SimpleFSDP(SequentialModel())
    
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
            logger.debug(f"Epoch {epoch}, Loss: {loss.item()}")
    
    assert isinstance(fsdp_model, SimpleFSDP)
    assert loss.item() > 0

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_sequential_model(rank, device)

# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 2 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 4 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 8 --oversubscribe python test/infra/distribution_strategy/test_fsdp.py
