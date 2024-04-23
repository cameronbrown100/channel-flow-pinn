# PINN for laminar channel flow
Example using Nvidia Modulus to simulate the laminar flow through a channel using a physics informed neural network (PINN) and transfer learning to different channel geometries.

# Contentss

# Setup
1. Install latest Windows graphics driver
2. Install and update WSL2 kernel https://docs.nvidia.com/cuda/wsl-user-guide/index.html
3. Install CUDA toolkit for WSL https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
4. Install Docker Desktop and Docker Engine
5. Docker command to run in same home directory
```
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -it --rm nvcr.io/nvidia/modulus/modulus:23.11 bash
```

