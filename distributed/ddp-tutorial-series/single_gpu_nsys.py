# -*- coding: utf-8 -*-
"""
Single-GPU transformer training with Nsight Systems (nsys) profiling support.

NVTX ranges label every phase of the training step so they appear as colored
bands in the Nsight Systems timeline:

    data-to-gpu  →  forward  →  loss  →  backward  →  optimizer

The cudaProfilerApi capture window skips the first --warmup-epochs epochs so
only steady-state iterations land in the .nsys-rep file.

── How to run ────────────────────────────────────────────────────────────────

  nsys profile \\
      --capture-range=cudaProfilerApi \\
      --capture-range-end=stop \\
      -t cuda,nvtx,osrt,cudnn,cublas \\
      -o profiles/single_gpu \\
      python single_gpu_nsys.py

  # Open the result:
  #   nsys-ui profiles/single_gpu.nsys-rep

See run_nsys_profile.sh for the full comparison workflow.
"""

import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_model import SmallTransformer, SyntheticTokenDataset

# Ampere+ GPUs benefit from TF32; harmless on older hardware
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        warmup_epochs: int,
        snapshot_path: str,
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.warmup_epochs = warmup_epochs
        self.snapshot_path = snapshot_path
        self.epochs_run = 0
        self.scaler = torch.cuda.amp.GradScaler()

        if os.path.exists(snapshot_path):
            self._load_snapshot(snapshot_path)

    def _load_snapshot(self, path: str):
        snapshot = torch.load(path, map_location=f"cuda:{self.gpu_id}", weights_only=True)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"[GPU{self.gpu_id}] Resumed from snapshot at epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int):
        snapshot = {"MODEL_STATE": self.model.state_dict(), "EPOCHS_RUN": epoch}
        torch.save(snapshot, self.snapshot_path)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | snapshot saved to {self.snapshot_path}")

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            torch.cuda.nvtx.range_push("forward")
            logits = self.model(source)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("loss")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        self.scaler.scale(loss).backward()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("optimizer")
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        torch.cuda.nvtx.range_pop()

        return loss.item()

    def _run_epoch(self, epoch: int):
        total_loss = 0.0
        for source, targets in self.train_data:
            torch.cuda.nvtx.range_push("data-to-gpu")
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("batch")
            total_loss += self._run_batch(source, targets)
            torch.cuda.nvtx.range_pop()

        steps = len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | loss {total_loss / steps:.4f}")

    def train(self, total_epochs: int):
        for epoch in range(self.epochs_run, total_epochs):
            if epoch == self.warmup_epochs:
                print(f"[GPU{self.gpu_id}] Warmup done — starting nsys capture")
                torch.cuda.cudart().cudaProfilerStart()

            torch.cuda.nvtx.range_push(f"epoch-{epoch}")
            self._run_epoch(epoch)
            torch.cuda.nvtx.range_pop()

            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)

        torch.cuda.cudart().cudaProfilerStop()
        print(f"[GPU{self.gpu_id}] nsys capture stopped")


def main(args):
    torch.manual_seed(args.seed)

    dataset = SyntheticTokenDataset(args.dataset_size, args.seq_len, args.vocab_size)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    model = SmallTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    trainer = Trainer(
        model=model,
        train_data=train_data,
        optimizer=optimizer,
        gpu_id=0,
        save_every=args.save_every,
        warmup_epochs=args.warmup_epochs,
        snapshot_path=args.snapshot_path,
    )
    trainer.train(args.total_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU transformer training with nsys profiling")

    # Training
    parser.add_argument("--total-epochs",   type=int,   default=5)
    parser.add_argument("--warmup-epochs",  type=int,   default=2,
                        help="Epochs to run before nsys capture starts")
    parser.add_argument("--save-every",     type=int,   default=5)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--snapshot-path",  type=str,   default="snapshot_single.pt")

    # Dataset
    parser.add_argument("--dataset-size",   type=int,   default=4096)
    parser.add_argument("--vocab-size",     type=int,   default=8192)
    parser.add_argument("--seq-len",        type=int,   default=128)

    # Model
    parser.add_argument("--d-model",        type=int,   default=512)
    parser.add_argument("--n-heads",        type=int,   default=8)
    parser.add_argument("--n-layers",       type=int,   default=6)

    args = parser.parse_args()
    main(args)
