# GCU-GS

GCU-GS is a research codebase for sparse-view Gaussian splatting. This repository currently provides the training and evaluation pipeline used for LLFF experiments.

> **Status:** This work is currently under submission, and the released code is an unpolished research snapshot that has not yet been fully cleaned.

## Installation

The code has been tested with the following environment:

- Ubuntu 20.04
- Python 3.9
- CUDA 11.8
- PyTorch 2.4.1

Make sure that CUDA 11.8 and a compatible C++ compiler are available before
building the CUDA extensions.

### 1. Create a Conda environment

```bash
conda create -n gcu-gs python=3.9 -y
conda activate gcu-gs
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

Install the CUDA 11.8 build of PyTorch 2.4.1:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Build and install PyTorch3D from source

Clone the official [PyTorch3D repository](https://github.com/facebookresearch/pytorch3d),
switch to its stable version, and install it from the local source tree:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout stable
pip install .
cd ..
```

### 5. Install the local CUDA extensions

Run the following commands from the repository root:

```bash
pip install ./submodules/diff-plane-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/test_uf
```


## Running

The LLFF dataset is used here as an example.

### Dataset structure

Arrange the dataset as follows:

```text
datasets/
└── LLFF/
    └── scene/
        ├── images/
        ├── sparse/
        └── sparse_views/
            └── fused.ply
```

Download the sparse-view `fused.ply` files from the
[FSGS data release](https://drive.google.com/drive/folders/1lYqZLuowc84Dg1cyb8ey3_Kb-wvPjDHA)
and place the corresponding file in each scene's `sparse_views/` directory.


```

### Train and evaluate

The default data root is configured in `scripts/train_llff.sh` as `BASE_DATA`

To use another dataset location without editing the script, override
`BASE_DATA` when launching it:

```bash
BASE_DATA=/path/to/datasets/LLFF bash scripts/train_llff.sh
```

The default experiment trains with three input views and geometric
densification:

```text
--n_views 3 --interval 10 --geom_densify
```

Start all LLFF scenes from the repository root:

```bash
bash scripts/train_llff.sh
```


## Acknowledgements

This project is built upon
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[FSGS](https://github.com/VITA-Group/FSGS),
[PGSR](https://github.com/zju3dv/PGSR),
We thank the authors for making their work publicly available.
