# ESXi 8.0 + Ubuntu 26.04 VM — GPU Passthrough Setup

## Overview

Running Ubuntu 26.04 LTS as a VM on ESXi 8.0 with the AMD Radeon RX 7900 XTX passed through
via VMDirectPath I/O. This gives native Linux ROCm support without WSL2 quirks.

```
ESXi 8.0 Host
├── Windows Server VM (management / Docker / general use)
└── Ubuntu 26.04 VM (ML training — GPU passed through)
    ├── ROCm 7.1.0 (native) or 7.2.4 (AMD repo)
    ├── Docker + Unsloth ROCm image
    └── Cognitive Core training pipeline
```

## Prerequisites

- ESXi 8.0 installed on the host
- AMD IOMMU / SVM enabled in BIOS
- 7900 XTX installed in the host (not yet used by any VM)
- Enough RAM to dedicate to the Ubuntu VM (16-32 GB recommended)

---

## Step 1: BIOS Configuration

Before anything else, ensure these are enabled in your motherboard BIOS:

```
AMD CBS → SVM Mode = Enabled          (AMD-V virtualization)
AMD CBS → IOMMU = Enabled             (I/O Memory Management Unit)
AMD CBS → ACS Enable = Enabled        (if available — helps with IOMMU groups)
```

Reboot after changing BIOS settings.

---

## Step 2: Enable Passthrough in ESXi

1. Log into the ESXi web UI (`https://<esxi-ip>/ui`)
2. Navigate to **Manage → Hardware → PCI Devices**
3. Find the AMD Radeon RX 7900 XTX (may show as multiple PCI devices):
   - `1002:744c` — GPU compute
   - `1002:7444` — Audio device (HDMI audio)
4. Select both devices and click **Toggle Passthrough**
5. The status should change to "Active" (green)
6. **Reboot the ESXi host** — this is required for passthrough to take effect

After reboot, verify in **Manage → Hardware → PCI Devices** that both show as "Active" under passthrough.

---

## Step 3: Create the Ubuntu 26.04 VM

### VM Settings

| Setting | Value |
|---|---|
| Name | `ubuntu-ml-training` |
| OS | Ubuntu Linux (64-bit) |
| Version | Ubuntu 26.04 LTS |
| CPU | 8 vCPU (16 threads) — the 7900X has 12 cores / 24 threads |
| RAM | 32 GB — reserve ALL (no swapping) |
| Disk | 200-300 GB (models + training data + containers + checkpoints) |
| Network | VM Network (same as host) |
| GPU | Passthrough device(s) — see below |

### Adding the GPU to the VM

1. Power off the VM (if running)
2. Edit VM Settings → **Add Other Device → PCI Device**
3. Select each passthrough device:
   - `1002:744c` (GPU)
   - `1002:7444` (Audio — optional, include for completeness)
4. **Important**: Set the GPU to use **PCIe passthrough**, not virtual

### CPU Configuration

If you want maximum training performance:

1. Edit VM Settings → CPU → **Hardware virtualization**
   - Expose hardware-assisted virtualization to guest OS: **Enable** (if available)
2. For CPU pinning (advanced):
   - In the VM's `.vmx` file, add:
   ```
   cpuid.coresPerSocket = "8"
   numvcpus = "8"
   ```

### Memory Reservation

**Critical**: Reserve all RAM for the VM. If ESXi swaps VM memory to disk,
GPU passthrough performance will tank.

1. Edit VM Settings → Memory → **Reserve all guest memory**
2. This means the RAM is permanently allocated to this VM
3. Ensure the host has enough RAM for ESXi itself (4-8 GB) + this VM

---

## Step 4: Install Ubuntu 26.04

1. Download the Ubuntu 26.04 LTS Server ISO from ubuntu.com
2. In the ESXi web UI, upload the ISO to a datastore
3. Mount the ISO to the VM and boot
4. Follow the installer — use the entire disk
5. Install OpenSSH server when prompted (for remote access)
6. Reboot and remove the ISO

Verify the GPU is visible:
```bash
lspci | grep -i amd
# Should show something like:
# 03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] (rev c8)
# 03:00.1 Audio device: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI/DP Audio
```

If the GPU shows up in `lspci` but the driver isn't loaded, that's expected —
ROCm will handle it.

---

## Step 5: Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

---

## Step 6: Install ROCm (Two Options)

### Option A: Ubuntu Native Packages (Easier, ROCm 7.1.0)

Ubuntu 26.04 ships ROCm in the standard repos. This is the simplest path:

```bash
# Install ROCm from Ubuntu repos
sudo apt install -y rocm-dev rocm-hip-sdk

# Add to PATH
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Set GPU architecture (CRITICAL for 7900 XTX)
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

### Option B: AMD Repository (ROCm 7.2.4 — Newer)

If you need the latest features or bug fixes:

```bash
# Add AMD's repo
sudo apt install -y wget gnupg2
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo "deb [arch=amd64] https://repo.radeon.com/rocm/7.2/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-dev

echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

### Verify ROCm

```bash
# Check ROCm installation
rocminfo | head -20

# Should show GPU info with gfx1100 (RDNA 3)
# Should see "Name: gfx1100" in the output
```

---

## Step 7: Launch the Training Docker Container

```bash
# Pull the Unsloth ROCm image (RDNA 3 / gfx1100 verified)
docker pull goldengrapegentleman/unsloth-rocm:2026.1.4-rocm7.1-gfx1100

# Launch with GPU access
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --shm-size=16g \
    -v /home/$USER/cognitive-core:/workspace \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    goldengrapegentleman/unsloth-rocm:2026.1.4-rocm7.1-gfx1100 \
    bash
```

**Note**: Adjust `-v` mount to wherever you cloned the project repo.

Inside the container, verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected:
# True
# AMD Radeon RX 7900 XTX
```

---

## Troubleshooting

### "No devices found" / GPU not visible in lspci
- Ensure passthrough is enabled and host was rebooted
- Check BIOS: SVM and IOMMU must be enabled
- Check IOMMU groups — if GPU shares a group with a critical device,
  you may need to use an IOMMU override in the VMX file

### "invalid device function" / HIP error
- Ensure `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set
- Check that the user is in the `video` and `render` groups:
  ```bash
  groups $USER
  # Should include 'video' and 'render'
  ```
- If not: `sudo usermod -aG video,render $USER` then log out/in

### VM won't boot with GPU passthrough
- Ensure RAM is fully reserved (not just allocated)
- Check that no other VM is using the same PCI device
- Try adding `hypervisor.cpuid.v0 = "FALSE"` to the VMX file

### Slow training / low GPU utilization
- Verify the GPU is actually being used (not CPU fallback):
  ```bash
  watch -n 1 rocm-smi
  ```
- Ensure `--shm-size=16g` is set on the Docker container
- Check that Docker isn't using a different runtime

---

## Comparison: WSL2 vs ESXi VM

| | WSL2 on Windows | ESXi VM |
|---|---|---|
| GPU access | Indirect (WSL2 + D3D12 driver) | Direct passthrough (native ROCm) |
| ROCm support | Limited, unofficial | Full, official packages |
| Stability | Occasional WSL2 quirks | Stable, production-grade |
| Resource control | Shares Windows resources | Dedicated CPU/RAM/disk |
| VM isolation | None — shares Windows kernel | Full VM isolation |
| Management | Windows Server | ESXi web UI + vCenter |
| Recommended for | Quick dev/testing | Serious training workloads |

**Verdict**: The ESXi VM approach is strictly better for this use case.
Native ROCm, dedicated resources, no WSL2 quirks.
