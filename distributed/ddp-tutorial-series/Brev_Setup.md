# DDP Profiling with Nsight Systems on Brev

## Prerequisites

- [Brev account](https://brev.dev) (free tier available)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) installed
  locally to view `.nsys-rep` profile files

---

## Step 1 — Launch the environment

Open the launchable link for this workshop. Select an **A100** instance with the
number of GPUs you want to profile (1, 2, or 4).

The base image is `nvcr.io/nvidia/pytorch:24.10-py3`, which includes:
- PyTorch 2.5 with CUDA 12.6
- Nsight Systems (`nsys`) pre-installed
- No additional setup required

Once the instance is running, open a terminal.

---

## Step 2 — Set up the repo

The `.brev/setup.sh` script runs automatically when the instance starts. Verify it
completed and navigate to the working directory:

```bash
cd ~/DDP_examples/distributed/ddp-tutorial-series
```

Run the smoke test to confirm everything works before profiling:

```bash
bash brev/smoke_test.sh
```

Expected output ends with `Smoke test complete — ready for profiling runs`.

---

## Step 3 — Run profiling jobs

```bash
# Single-GPU baseline
bash brev/run_single_gpu.sh

# Multi-GPU DDP (uses all available GPUs by default, or specify a count)
bash brev/run_ddp.sh 2
bash brev/run_ddp.sh 4
```

Profiles are written to `~/ddp_profiles/`.

---

## Step 4 — Download and open profiles

In a local terminal:

```bash
scp -i ~/.ssh/brev <instance-user>@<instance-ip>:~/ddp_profiles/*.nsys-rep ~/Downloads/

nsys-ui ~/Downloads/<profile>.nsys-rep
```

The instance IP and SSH key path are shown in the Brev dashboard under **Connect**.
