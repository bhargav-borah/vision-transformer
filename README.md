# Vision Transformer for Image Classification

This repository implements a Vision Transformer (ViT) for image classification on multiple datasets, including Flowers102, CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST. The implementation is based on the paper _"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"_ by Dosovitskiy et al. (2020).

## Overview

This project provides a PyTorch implementation of a Vision Transformer (ViT-B/16) trained from scratch on five image classification datasets. It supports both color and grayscale images, includes data augmentation, and evaluates performance using loss, accuracy, F1 score, and recall. The code is available as both a command-line script (`train_vit.py`) and a Jupyter notebook (`train_vit_notebook.ipynb`) for flexibility.

## Features

- **Multi-Dataset Support**: Flowers102, CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST
- **Flexible Input Handling**: Color and grayscale image support
- **Data Augmentation**: Random resized crops, horizontal flips, and color jitter
- **Comprehensive Evaluation**: Tracks loss, accuracy, F1 score, and recall
- **Reproducible Workflow**: Weight initialization, random seed setting, and checkpointing
- **Output Generation**: Saves weights, metrics, and plots for analysis

## Datasets

| Dataset        | Classes | Image Type | Train | Val | Test  |
|----------------|---------|------------|--------|-----|--------|
| Flowers102     | 102     | Color      | ~1,020 | ~1,020 | ~6,149 |
| CIFAR-10       | 10      | Color      | 50,000 | 5,000 | 5,000 |
| CIFAR-100      | 100     | Color      | 50,000 | 5,000 | 5,000 |
| MNIST          | 10      | Grayscale  | 60,000 | 5,000 | 5,000 |
| Fashion-MNIST  | 10      | Grayscale  | 60,000 | 5,000 | 5,000 |

> Note: For datasets without predefined validation sets, 50% of the test set is used for validation.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (with GPU support recommended)
- ~3.5 GB of disk space for datasets

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bhargav-borah/vision-transformer.git
   cd vision-transformer
````

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Script

Run the training script with desired arguments:

```bash
python train_vit.py --dataset flowers102 --num_epochs 300 --lr 3e-4 --batch_size 32
```

**Supported Datasets**:

* `flowers102`
* `cifar10`
* `cifar100`
* `mnist`
* `fashionmnist`

**Key Arguments**:

* `--dataset`: Dataset to train on (default: `flowers102`)
* `--num_epochs`: Number of epochs (default: `300`)
* `--lr`: Learning rate (default: `3e-4`)
* `--batch_size`: Batch size (default: `32`)

### Jupyter Notebook

1. Start the notebook:

   ```bash
   jupyter notebook
   ```
2. Open `train_vit_notebook.ipynb`
3. Modify the main function to select a dataset:

   ```python
   main('cifar10')
   ```
4. Run all cells to train and evaluate the model.

## Outputs

* **Model Checkpoint**:

  * `data/best_vit_<dataset>.pth`

* **Metrics**:

  * `data/metrics_<dataset>.json`

* **Visualizations**:

  * Batch example: `data/batch_visualization_<dataset>.png`
  * Training plots: `data/training_metrics_<dataset>.png`

### Example

Train on CIFAR-10 for 50 epochs:

```bash
python train_vit.py --dataset cifar10 --num_epochs 50
```

Expected output files in `data/`:

* `best_vit_cifar10.pth`
* `metrics_cifar10.json`
* `batch_visualization_cifar10.png`
* `training_metrics_cifar10.png`

## Citation

This implementation is based on the Vision Transformer architecture introduced in:

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszoreit, J., & Houlsby, N. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv preprint [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).

