# Testcrypt — AES-CTR Encryption (CPU + CUDA GPU Ready)

This project implements AES-CTR encryption with:
- **CPU fallback** (OpenSSL AES-CTR)
- **Optional GPU acceleration** (CUDA AES kernel)

The build system automatically detects CUDA:
- If CUDA is installed → **GPU mode enabled**
- If CUDA is NOT installed → **CPU-only mode**

---

## Running on a Machine WITHOUT CUDA (CPU Only)

No NVIDIA GPU required.

### Steps:

```bash
git clone <repo_url>
cd Testcrypt
mkdir build
cd build
cmake ..
make -j
./testAES
```

### Expected Output:

```
CPU MODE (GPU not available)
AES TEST PASSED 
```

---

## Running on a Machine WITH NVIDIA GPU (CUDA Enabled)

Follow these steps on a GPU system.

### 1️ Install CUDA Toolkit

```bash
sudo apt install nvidia-cuda-toolkit
```

Or install from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads) for best performance.

**Verify CUDA installation:**

```bash
nvcc --version
```

### 2️ Add a Real AES CUDA Kernel

Open the file:

```
gpu/AESCudaAdapter.cu
```

Inside this file, replace the placeholder CUDA kernel call with an actual AES CUDA kernel from one of these repositories:

- https://github.com/cihangirtezcan/CUDA_AES
- https://github.com/gh0stintheshe11/CUDA-Accelerated-AES-Encryption

Your kernel function must match this call:

```cpp
aes_encrypt_ctr_gpu<<<grid, block>>>(dOut, dIn, key, iv, nBlocks);
```

**Copy the `.cu` and `.h` kernel files** from the chosen repo into:

```bash
Testcrypt/gpu/
```

### 3️ Build and Run (GPU Mode)

```bash
git clone <repo_url>
cd Testcrypt
mkdir build
cd build
cmake ..
make -j
./testAES
```

### Expected Output on a CUDA System:

```
CUDA Found Enabling GPU Mode
GPU FOUND
AES GPU TEST PASSED 
```

## Troubleshooting

### CUDA Not Detected

If you have CUDA installed but CMake doesn't detect it:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake ..
```

### Compilation Errors

Make sure you have:
- OpenSSL development libraries: `sudo apt install libssl-dev`
- C++ compiler with C++11 support
- CMake 3.10 or higher

---
