# DDP Profiling with Nsight Systems on Brev

## Prerequisites

- Brev launchable using the **L40S 48GB, 2-GPU** instance
- NVIDIA Nsight Systems installed locally to view `.nsys-rep` profile files

---

## Step 1 — Launch the environment

Open the launchable link for this workshop and select the **L40S 48GB, 2-GPU**
instance. Use 256 GB of disk storage.

The launchable uses `nvcr.io/nvidia/pytorch:24.10-py3`, which includes:

- PyTorch 2.5 with CUDA 12.6
- Nsight Systems (`nsys`)
- NCCL and `torchrun`

If the selected image does not already include PyTorch, `.brev/setup.sh`
installs a CUDA-enabled PyTorch wheel.

Once the instance is running, open a terminal.

---

## Step 2 — Run setup

Once the instance is running, open a terminal and go to the workshop directory:

```bash
cd ~/DDP_examples/distributed/ddp-tutorial-series
```

Run the Brev setup script:

```bash
bash .brev/setup.sh
```

This installs or verifies Nsight Systems and PyTorch, then creates the profile
output directory.

Run the smoke test before profiling:

```bash
bash brev/smoke_test.sh
```

Expected output ends with:

```text
Smoke test complete — ready for profiling runs
```

---

## Step 3 — Run profiling jobs

```bash
# Single-GPU baseline
bash brev/run_single_gpu.sh

# 2-GPU DDP
bash brev/run_ddp.sh
```

`run_ddp.sh` uses all visible GPUs by default. On the L40S 2-GPU instance this
is equivalent to:

```bash
bash brev/run_ddp.sh 2
```

Profiles are written to:

```text
~/ddp_profiles/
```

---

## Step 4 — Download and open profiles

In Jupyter, open the file browser and go to:

```text
/home/ubuntu/ddp_profiles
```

Right-click the `.nsys-rep` files and select **Download**.

Open the downloaded files locally with NVIDIA Nsight Systems.

---

## What to look for

- Single GPU: one process, no NCCL communication
- 2-GPU DDP: two worker processes, one per L40S
- NVTX ranges: `forward`, `loss`, `backward`, `optimizer`
- NCCL AllReduce kernels inside the `backward` range
