import torch
import random
import numpy as np
from huggingface_hub import hf_hub_download

def get_batch(dataset, batch_size: int, context_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ds = torch.from_numpy(dataset).astype(np.int64)
    num_data = ds.shape[-1]
    train = torch.empty(batch_size, context_len,device=device)
    target = torch.empty(batch_size, context_len, device=device)
    for i in range(batch_size):
        start_idx = random.randint(0,num_data-context_len-1)
        train[i] = ds[start_idx:start_idx+context_len]
        target[i] = ds[start_idx+1:start_idx+context_len+1]
    return (train,target)

class MemoryMappedDataLoader:
    def __init__(self, ds_path: str, batch_size: int, context_len: int, device: str):
        repo, filename = ds_path.split(":")
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
        )
        self.ds = np.memmap(path, dtype=np.uint16, mode="r")

        print(f"Loading dataset with {len(self.ds)} samples")
        self.batch_size = batch_size
        self.context_len = context_len
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        start_indices = torch.randint(len(self.ds) - self.context_len, (self.batch_size, ))
        train = torch.stack([ torch.from_numpy(self.ds[i:i+self.context_len].astype(np.int64)) for i in start_indices])
        pred = torch.stack([ torch.from_numpy(self.ds[i+1:i+self.context_len+1].astype(np.int64)) for i in start_indices])
        train = train.to(self.device)
        pred = pred.to(self.device)
        return (train, pred)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    state = {
        "it": iteration,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    torch.save(state, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    return int(state["it"])
