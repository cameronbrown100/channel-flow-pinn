# PINN for laminar channel flow
Example using Nvidia Modulus to simulate the laminar flow through a channel using a physics informed neural network (PINN) and transfer learning to different channel geometries.
![Validation of u velocity](https://github.com/cameronbrown100/channel-flow-pinn/blob/main/example_validator_u.png)

# Contents

# Setup
1. Install latest Windows graphics driver
2. Install and update WSL2 kernel https://docs.nvidia.com/cuda/wsl-user-guide/index.html
3. Install CUDA toolkit for WSL https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
4. Install Docker Desktop and Docker Engine
5. Docker command to run in same home directory
```
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -it --rm nvcr.io/nvidia/modulus/modulus:23.11 bash
```
Modulus documentation:
https://docs.nvidia.com/deeplearning/modulus/getting-started/index.html

Modulus packages:
https://github.com/NVIDIA/modulus-sym/tree/main?tab=readme-ov-file

# Running
## Geometries
### Baseline
![Baseline geometry used to train initial model](https://github.com/cameronbrown100/channel-flow-pinn/blob/main/baseline_geometry.png | width=200)

### Target
![Geometry target of transfer learning](https://github.com/cameronbrown100/channel-flow-pinn/blob/main/target_geometry.png | width=200)

1. Run script channel_flow_baseline.py. Trains the model on a geometry with equal sized constrictions.
3. Model is saved to outputs/channel_flow_baseline/baseline
4. Run script channel_flow_target.py. Applies transfer learning train the model on a geometry with unequal sized constrictions.
5. Model is saved to outputs/channelo_flow_target/target
