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

## Step 2 — Set up the repo

The `.brev/setup.sh` script runs automatically when the instance starts. If you
need to run it manually:

```bash
bash .brev/setup.sh
```

Navigate to the workshop directory:

```bash
cd ~/DDP_examples/distributed/ddp-tutorial-series
```

Run the smoke test:

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

In a local terminal:

```bash
scp -i ~/.ssh/brev <instance-user>@<instance-ip>:~/ddp_profiles/*.nsys-rep ~/Downloads/

nsys-ui ~/Downloads/<profile>.nsys-rep
```

The instance IP and SSH key path are shown in the Brev dashboard under
**Connect**.

---

## What to look for

- Single GPU: one process, no NCCL communication
- 2-GPU DDP: two worker processes, one per L40S
- NVTX ranges: `forward`, `loss`, `backward`, `optimizer`
- NCCL AllReduce kernels inside the `backward` range

The L40S 2-GPU instance will not reproduce the full Tillicum 1/2/4/8 scaling
study, but it is enough to show the DDP mechanics and NCCL overlap clearly.
