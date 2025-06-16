# MATLAB-Traffic-Data-Analysis-for-Modelling-and-Prediction-Project

## Quick start

> **MATLAB R2022b +** Deep Learning Toolbox & Parallel Computing Toolbox (GPU optional)

```matlab
git clone https://github.com/BarbarosTeoman/MATLAB-Traffic-Data-Analysis-for-Modelling-and-Prediction-Project.git
cd MATLAB-Traffic-Data-Analysis-for-Modelling-and-Prediction-Project

% Pull large *.mat files handled by Git LFS
git lfs pull
````

1. **Open MATLAB** in the project root.
2. Run **`main.m`**

---

## Repository layout

```
.
├── ngsim_cached_data.mat    # binary cache (LFS)
├── TrainDS.mat              # Training data
├── ValDS.mat                # Validation data (LFS)
├── Test.mat                 # Test data
├── trainedNetwork.mat       # saved network after training
├── trainingInfo.mat         # saved training information after training
├── LICENSE
└── README.md
```

> Large files (`*.mat` > 100 MB) are tracked with **Git LFS**.
> Install Git LFS (`https://git-lfs.github.com`) before cloning.

---

## Training details

| Parameter        | Value                            |
| ---------------- | -------------------------------- |
| Epochs           | 10 (early-stops on Val loss)     |
| Batch size       | 256                              |
| Optimizer        | Adam (1e-3, polynomial decay)    |
| CNN filter sizes | 3×3 @ 32, 64                     |
| BiLSTM units     | 64 (occupancy) + 32 (kinematics) |
| Output           | (X, Y, Speed) 30 frames ahead    |

---

## Dataset citation

NGSIM trajectory data courtesy of the **U.S. Federal Highway Administration**
[https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)

If you use this code, please cite both NGSIM and this repository.

---

## License

This project is released under the **MIT License** (see `LICENSE`).
