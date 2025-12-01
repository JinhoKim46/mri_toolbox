
# MRI Toolbox <p align="left">
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>

  <!-- Python Version -->
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">


  <!-- PyPI Downloads -->
  <a href="https://pypi.org/project/mri-toolbox/">
    <img src="https://img.shields.io/pypi/dm/mri-toolbox.svg" alt="PyPI downloads">
  </a>

  <!-- Issues -->
  <a href="https://github.com/JinhoKim46/mri_toolbox/issues">
    <img src="https://img.shields.io/github/issues/JinhoKim46/mri_toolbox.svg" alt="Issues">
  </a>

  <!-- Stars -->
  <a href="https://github.com/JinhoKim46/mri_toolbox/stargazers">
    <img src="https://img.shields.io/github/stars/JinhoKim46/mri_toolbox.svg" alt="Stars">
  </a>

  <!-- Last Commit -->
  <img src="https://img.shields.io/github/last-commit/JinhoKim46/mri_toolbox.svg" alt="Last Commit">
</p>

---

## Overview

**MRI Toolbox** is a Python package for MRI reconstruction and visualization.  
It provides reference implementations of SENSE, GRAPPA, and compressed sensing algorithms, as well as a flexible visualization interface designed for MRI data exploration and quantitative evaluation.

---

## Features

### 1. MRI Reconstruction
Reference implementations for classical and widely used MRI reconstruction methods:

- **SENSE (Sensitivity Encoding)**  
  Parallel imaging reconstruction with **SigPy** and **CuPy** acceleration.

- **GRAPPA**  
  GRAPPA reconstruction using the **PyGRAPPA** library.

- **Compressed Sensing MRI**  
  Basic ℓ1-regularized iterative reconstruction built on **SigPy/CuPy**.

---

### 2. Advanced Visualization (`imshow`)
The `toolbox.visualizaing.imshow` module provides a powerful MRI viewer with:

- Side-by-side comparison of multiple reconstructions  
- Ground truth comparison  
- MIP visualization for 3D volumes  
- Extensible hook system:
  - PSNR, SSIM overlays  
  - Bounding boxes  
  - Line profiles  
  - Custom user-defined hooks  

---

## Installation
1. Clone the repository
   ```bash
   git clone git@github.com:JinhoKim46/mri_toolbox.git
   cd mri_toolbox
   ```
2. Create the conda environment:
   ```bash
   conda create -n mri_toolbox python=3.10.14
   conda activate mri_toolbox
   ```
3. Install PyTorch (CUDA 11.8)
   ```bash
      pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install the toolbox
   ```bash
      pip install -e .
   ```
---

## Usage
### Demonstration Notebooks
Check the `Demo/` directory for Jupyter Notebooks demonstrating how to use the toolbox. These notebooks cover:
- Basic usage of `imshow` for visualization.
- Running `SENSE` reconstruction.
- Applying predefined visualization hooks.

---
### Data
  <!-- Zenodo DOI (placeholder — replace with your Zenodo DOI) -->
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13912092.svg)](https://doi.org/10.5281/zenodo.13912092)
Demo data can be downloaded from Zenodo. 
Place the data in the `Data/` directory or update paths in the notebooks accordingly.

---
## Project Structure
```
mri_toolbox/
├── Data/                   # Directory for dataset files (e.g., .h5)
├── Demo/                   # Jupyter notebooks for demonstration
├── toolbox/                # Main package source code
│   ├── reconstructions.py  # SENSE, GRAPPA implementations
│   ├── visualizaing.py     # imshow and plotting tools
│   ├── utils.py            # Helper functions and metrics
│   └── imshow_hooks/       # Visualization hooks (metrics, boxes, etc.)
├── pyproject.toml          # Project configuration
├── LICENSE                 # License information
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---
## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
## Acknowledgements
This toolbox uses the following open-source projects to support MRI reconstruction:
- SigPy (MIT License) — a comprehensive framework for iterative MRI reconstruction
  - https://github.com/mikgroup/sigpy

- PyGRAPPA (MIT License) — a Python implementation of GRAPPA for parallel MRI
  - https://github.com/mriphysics/pygrappa

Their contributions to the scientific and open-source community are gratefully acknowledged.

---
## Author

**Jinho Kim**  
Email: jinho.kim@fau.de

If this toolbox is useful in your research, please consider citing the repository.