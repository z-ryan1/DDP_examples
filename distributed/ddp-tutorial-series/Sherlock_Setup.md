# DDP Profiling with Nsight Systems on Sherlock

Uses Apptainer + NGC PyTorch container on the `serc` partition. The container
ships with PyTorch and `nsys` pre-installed, sidestepping module compatibility
issues between `gpu` and `serc` partitions.

## Prerequisites

- Sherlock access with `serc` partition membership (Stanford SUNet ID)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) installed
  locally to view `.nsys-rep` profile files

---

## Step 1 — Connect and clone

Clone into `$SCRATCH` — Sherlock recommends scratch for active work (`$HOME` has a 15 GB limit).

```bash
ssh SUNetID@sherlock.stanford.edu

git clone https://github.com/z-ryan1/DDP_examples.git $SCRATCH/DDP_examples
cd $SCRATCH/DDP_examples/distributed/ddp-tutorial-series
```

---

## Step 2 — Submit jobs

The first job to run will pull the NGC PyTorch container (~8 GB) to
`$SCRATCH/pytorch-24.10.sif`. Subsequent jobs reuse it.

```bash
sbatch slurm/smoke_test_sherlock.sh

sbatch slurm/single_gpu_nsys_sherlock.sh

./slurm/submit_ddp_sherlock.sh 2
./slurm/submit_ddp_sherlock.sh 4
./slurm/submit_ddp_sherlock.sh 8
```

Profiles are written to `$SCRATCH/ddp_profiles/`.

---

## Step 3 — Download and open profiles

```bash
scp SUNetID@login.sherlock.stanford.edu:$SCRATCH/ddp_profiles/*.nsys-rep ~/Downloads/

nsys-ui ~/Downloads/<profile>.nsys-rep
```
