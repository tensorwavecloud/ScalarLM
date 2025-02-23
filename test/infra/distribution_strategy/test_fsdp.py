import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
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
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)
    
    fsdp_model = SimpleFSDP(model)
    
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        input_data = torch.randn(32, 10, device=device)
        output = fsdp_model(input_data)
        
        target = torch.randn(32, 5, device=device)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        fsdp_model.backward(loss)
        optimizer.step()
        
        if rank == 0:
            logger.debug(f"Epoch {epoch}, Loss: {loss.item()}")
    
    assert isinstance(fsdp_model, SimpleFSDP)
    assert loss.item() > 0

def test_transformer_model(comm, rank, world_size, device):
    ntokens = 1000
    emsize = 64
    d_hid = 64
    nlayers = 1
    nhead = 2
    dropout = 0.2

    model = SimpleTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    for param in model.parameters():
        comm.Bcast(param.data.numpy(), root=0)

    model = SimpleFSDP(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def create_dataset(num_samples, seq_length):
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        data = torch.randint(0, ntokens, (num_samples, seq_length), dtype=torch.long)
        target = torch.randint(0, ntokens, (num_samples, seq_length), dtype=torch.long)
        torch.set_rng_state(rng_state)
        return TensorDataset(data, target)

    num_samples = 1000
    seq_length = 20
    batch_size = 16
    dataset = create_dataset(num_samples, seq_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    def train(model, data_loader, optimizer, criterion, epochs, device):
        model.train()
        total_loss = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_loss = 0

            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.view(-1, ntokens), target.view(-1))

                model.backward(loss)
                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0 and rank == 0:
                    logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Synchronize epoch_loss across all ranks
            epoch_loss = comm.allreduce(epoch_loss, op=MPI.SUM)
            if rank == 0:
                avg_loss = epoch_loss / (len(data_loader) * world_size)
                logger.debug(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            total_loss += epoch_loss

        return total_loss

    total_loss = train(model, train_loader, optimizer, criterion, epochs=5, device=device)

    comm.Barrier()
    if rank == 0:
        logger.debug("Training complete.")

    assert isinstance(model, SimpleFSDP)
    assert total_loss > 0

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_sequential_model(rank, device)
    test_transformer_model(comm, rank, world_size, device)


# PYTHONPATH=/app/cray/ mpirun --allow-run-as-root -np 2 --oversubscribe python test_fsdp.py