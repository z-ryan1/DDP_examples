# DDP Profiling with Nsight Systems on Tillicum

## Prerequisites

- Tillicum access (UW NetID + Duo 2FA)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) installed
  locally to view `.nsys-rep` profile files

---

## Step 1 — Connect and clone

```bash
ssh UWNetID@tillicum.hyak.uw.edu

git clone https://github.com/z-ryan1/DDP_examples.git
cd DDP_examples/distributed/ddp-tutorial-series
```

---

## Step 2 — Verify PyTorch installation (must be on a compute node)

PyTorch with CUDA must be run on a GPU node, not the login node.

```bash
salloc --qos=interactive --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=02:00:00

module load conda
conda activate pytorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.6.x+cu124 / True

exit
```

### Optional: Create your own conda environment for PyTorch

```bash
module load conda
conda create --prefix <path_to_your_environment> python=3.11 -y
```
> **WARNING:** Conda environment and package locations
> - The environment shoule live in project/lab dedicated storage `/gpfs/projects/` which has 1 TB of shared storage. 
> - Avoid `$HOME` (10 GB limit) and `/gpfs/scrubbed/` for long-term storage (auto-deletes after 60 days of inactivity).

GPU-aware PyTorch must be installed on a GPU node, not the login node which is CPU-only.

```bash
salloc --qos=interactive --gres=gpu:1 --cpus-per-task=8 --mem=200G --time=02:00:00

module load conda
conda activate <path_to_your_environment>
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.6.x+cu124 / True

exit
```

---

## Step 3 — Smoke test (login node)

```bash
export CONDA_ENV=/gpfs/software/miniforge3/25.3.1-3/envs/pytorch

sbatch slurm/smoke_test.sh
squeue -u $USER
cat slurm-<jobid>.out    # confirm both tests show PASSED
```

Clean up the snapshots saved from the test:

```bash
rm snapshot_single.pt snapshot_ddp.pt
```

---

## Step 4 — Submit profiling jobs

```bash
export CONDA_ENV=/gpfs/software/miniforge3/25.3.1-3/envs/pytorch

./slurm/submit_single_gpu.sh

./slurm/submit_ddp.sh 2
./slurm/submit_ddp.sh 4
./slurm/submit_ddp.sh 8
```

Profiles are written to `/gpfs/scrubbed/$USER/ddp_profiles/`.

---

## Step 5 — Download and open profiles

```bash
scp UWNetID@tillicum.hyak.uw.edu:/gpfs/scrubbed/$USER/ddp_profiles/*.nsys-rep ~/Downloads/

nsys-ui ~/Downloads/<profile>.nsys-rep
```
Or directly open the profiles in NVIDIA Nsight Systems application.