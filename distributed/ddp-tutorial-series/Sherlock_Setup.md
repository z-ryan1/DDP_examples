# DDP Profiling with Nsight Systems on Sherlock

## Prerequisites

- Sherlock access (Stanford SUNet ID)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) installed
  locally to view `.nsys-rep` profile files

---

## Step 1 — Connect and clone

```bash
ssh SUNetID@sherlock.stanford.edu

git clone https://github.com/z-ryan1/DDP_examples.git
cd DDP_examples/distributed/ddp-tutorial-series
```

---

## Step 2 — Verify the environment (compute node)

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00

module load py-pytorch/2.4.1_py312
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.4.1+cu... / True

exit
```

> No environment setup needed — Sherlock provides a pre-built PyTorch module.

---

## Step 3 — Smoke test

Delete any stale snapshots, then submit:

```bash
rm -f snapshot_single.pt snapshot_ddp.pt

sbatch slurm/smoke_test_sherlock.sh
squeue -u $USER
cat slurm-<jobid>.out    # confirm both tests show PASSED
```

---

## Step 4 — Submit profiling jobs

```bash
./slurm/submit_single_gpu_sherlock.sh

./slurm/submit_ddp_sherlock.sh 2
./slurm/submit_ddp_sherlock.sh 4
./slurm/submit_ddp_sherlock.sh 8
```

Profiles are written to `$SCRATCH/ddp_profiles/`.

---

## Step 5 — Download and open profiles

```bash
scp SUNetID@login.sherlock.stanford.edu:$SCRATCH/ddp_profiles/*.nsys-rep ~/Downloads/

nsys-ui ~/Downloads/<profile>.nsys-rep
```
