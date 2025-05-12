# Vision Transformer for Image Classification

This repository implements a Vision Transformer (ViT) for image classification on multiple datasets, including Flowers102, CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST. The implementation is based on the paper *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* by Dosovitskiy et al. (2020).

## Features
- Supports training ViT from scratch on five datasets.
- Handles color (Flowers102, CIFAR-10, CIFAR-100) and grayscale (MNIST, Fashion-MNIST) images.
- Includes data augmentation, training, validation, and test evaluation.
- Saves model checkpoints, metrics, and visualization plots.

## Datasets
- **Flowers102**: 102 classes, ~1,020 train, ~1,020 val, ~6,149 test.
- **CIFAR-10**: 10 classes, 50,000 train, 5,000 val, 5,000 test.
- **CIFAR-100**: 100 classes, 50,000 train, 5,000 val, 5,000 test.
- **MNIST**: 10 classes, 60,000 train, 5,000 val, 5,000 test.
- **Fashion-MNIST**: 10 classes, 60,000 train, 5,000 val, 5,000 test.

Validation sets for CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST are created by splitting the test set (50%/50%).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhargav-borah/vision-transformer.git
   cd vision-transformer