# MATLAB-Traffic-Data-Analysis-for-Modelling-and-Prediction-Project

>This project was developed for **MYZ 307E – Machine Learning for Electrical and Electronics Engineering**  
(CRN 22220), Istanbul Technical University, Spring 2025.

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

> ⚠️ **Note**: The project will automatically **load pre-processed `.mat` files** (TrainDS, ValDS, Test, and the trained network). This avoids the need to re-run the full data preprocessing and training pipeline, which can be time-consuming.

---

## Repository layout

```
.
├── main.m                      # Main MATLAB file
├── ngsim_cached_data.mat       # binary cache (LFS)
├── neighInputOdd.mat           # Preprocessed LSTM Input (LFS)
├── targetInputOddCentered.mat  # Preprocessed LSTM Input (LFS)
├── targetInputEvenCentered.mat # Preprocessed LSTM Input (LFS)
├── TrainDS.mat                 # Preprocessed training dataset
├── ValDS.mat                   # Preprocessed validation dataset (LFS)
├── Test.mat                    # Preprocessed test dataset
├── trainedNetwork.mat          # Pre-trained network
├── trainingInfo.mat            # Training log and metrics
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

> Training is skipped by default — the pretrained model is automatically loaded unless deleted.

---

## Dataset citation

NGSIM trajectory data courtesy of the **U.S. Federal Highway Administration**
[https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)

If you use this code, please cite both NGSIM and this repository.

---

## License

This project is released under the **MIT License** (see `LICENSE`).
