---
title: "Distributed Training Basics"
description: "Multi-GPU and multi-node training with PyTorch Distributed Data Parallel."
date: "2026-06-06"
tags: ["deep-learning", "distributed-training", "multi-gpu"]
---

Distributed training spreads computation across multiple GPUs or machines, enabling faster training and larger models.

## Data Parallel (Single Machine, Multiple GPUs)

```python
import torch.nn as nn
import torch.distributed as dist

# Wrap model with DataParallel (simple but less efficient)
model = nn.DataParallel(model)
model = model.cuda()

# Training loop remains the same
for inputs, targets in train_loader:
    inputs = inputs.cuda()
    targets = targets.cuda()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Distributed Data Parallel (DDP)

More efficient than DataParallel:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Setup
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Main training function
def main(rank, world_size, args):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4
    )
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffle
        
        for inputs, targets in train_loader:
            inputs = inputs.cuda(rank)
            targets = targets.cuda(rank)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    cleanup()


# Launch with torchrun
# torchrun --nproc_per_node=8 train.py
```

## Distributed Training with Launch Script

```python
#!/usr/bin/env python3
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def run_worker(rank, world_size, args):
    # Setup
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Create model (with sync batch norm for DDP)
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.SyncBatchNorm(64),  # Synchronize batch norm stats
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).cuda(rank)
    
    model = DDP(model, device_ids=[rank])
    
    # Rest of training...
    ...

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size)
```

## Saving and Loading Checkpoints

```python
# Save (only on rank 0)
def save_checkpoint(model, optimizer, epoch, path):
    if dist.get_rank() == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, path)

# Load (all ranks)
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=f'cuda:{dist.get_rank()}')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']
```

## Gradient Accumulation in DDP

```python
def train_ddp(model, train_loader, optimizer, epoch, accumulation_steps=4):
    model.train()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        # Sync gradients every accumulation step
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Multi-Node Training

```python
# Node 0 (master)
#   python train.py --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1

# Node 1
#   python train.py --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1


# Modified main for multi-node
def main(args):
    world_size = args.nnodes * args.nproc_per_node
    mp.spawn(
        run_worker,
        args=(world_size, args),
        nprocs=args.nproc_per_node,
        join=True
    )
```

## Performance Tips

| Technique | Benefit |
| --- | --- |
| DDP over DataParallel | Lower communication overhead |
| Use NCCL backend | GPU-to-GPU communication |
| Gradient compression | Reduce bandwidth (with accuracy trade-off) |
| CUDA graphs | Reduce kernel launch overhead |

Distributed training is essential for training large models efficiently.