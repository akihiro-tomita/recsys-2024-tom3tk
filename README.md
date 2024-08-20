# Recsys Challenge 2024 - Team Tom3TK

This repository contains Team Tom3TK's code submission for the Recsys 2024 Challenge. Our solution is implemented using Jupyter Notebooks and executed in the official Kaggle Docker environment.

## Our Approach

For a detailed explanation of our methodology, please refer to our paper: [Paper link to be added]

## Repository Structure

```
.
├── 0.precompute/
│   ├── a. get_article_embedding.ipynb
│   ├── b. item2vec.ipynb
│   ├── c. inview_occur.ipynb
│   └── d. article_pop_inview.ipynb
├── 1.feature_engineering/
│   └── feature-engineering.ipynb
├── 2.train_inference/
│   ├── a. train.ipynb
│   ├── b. inference.ipynb
│   └── c. create_submission_file.ipynb
└── 3.ablation/
    ├── 1.leaky/
    ├── 2.embedding/
    └── 3.dataset_size/
```

### 0. Precompute
- `get_article_embedding.ipynb`: Generates latent representations of articles using 'multilingual-e5-large'.
- `item2vec.ipynb`: Applies word2vec to learn article representations based on their appearance in user history.
- `inview_occur.ipynb`: Calculates and counts co-occurrences of articles in the in-view section.
- `article_pop_inview.ipynb`: Computes the frequency of each article appearing in-view over time.

### 1. Feature Engineering
- `feature-engineering.ipynb`: Calculates features from train/valid/test behaviors and user history.

### 2. Train/Inference
- `train.ipynb`: Trains 8 LightGBM models with LambdaRank using different data chunks, random states, and epochs.
- `inference.ipynb`: Generates predictions for the test dataset.
- `create_submission_file.ipynb`: Prepares the final submission file for Codabench.

### 3. Ablation Studies
- `1.leaky/`: Investigates the impact of removing potentially leaky features.
- `2.embedding/`: Compares different embedding models for article representation.
- `3.dataset_size/`: Examines the effect of dataset size on model performance.

## Reproduction Guide

Follow these steps to replicate our environment and results:

1. Pull the Kaggle Docker image:
   ```
   docker pull gcr.io/kaggle-gpu-images/python:v126
   ```
   Note: We tested with v126, but recent versions should be compatible.

2. Launch a Docker container:
   ```
   docker run --gpus all -it --rm \
     -p 8888:8888 \
     -v /path/to/your/local/directory:/home/ \
     gcr.io/kaggle-gpu-images/python:v126
   ```

3. Install Polars (if not pre-installed):
   ```
   pip install polars==0.18.4
   ```
   Note: Use version 0.x.x due to breaking changes in 1.0.0+.

4. Start Jupyter Lab:
   ```
   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
   ```

5. Execute the notebooks in the following order:
   - Run notebooks in the `0.precompute/` directory. Save results in a designated directory (e.g., `/home/data/`).
   - Run the feature engineering notebook in `1.feature_engineering/`.
   - Run the training and inference notebooks in `2.train_inference/`.

## Requirements
- Linux/Ubuntu environment
- GPU (required for embedding computations)
- Docker
- CUDA-compatible environment

