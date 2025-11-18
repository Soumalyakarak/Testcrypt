# Testcrypt — AES-CTR Encryption (CPU + CUDA GPU Ready)

This project implements AES-CTR encryption with:
- **CPU fallback** (OpenSSL AES-CTR)
- **Optional GPU acceleration** (CUDA AES kernel)

The build system automatically detects CUDA:
- If CUDA is installed → **GPU mode enabled**
- If CUDA is NOT installed → **CPU-only mode**

---
## Running on a Machine WITH NVIDIA GPU (CUDA Enabled)

Follow these steps on a GPU system.

### Install CUDA Toolkit

```bash
sudo apt install nvidia-cuda-toolkit
```

Or install from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads) for best performance.

**Verify CUDA installation:**

```bash
nvcc --version
```

### Build and Run (GPU Mode)

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
