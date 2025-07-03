# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

Official implementation of "ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection" (IEEE TIFS 2023).

## Features

- **Decomposed Spatial-Temporal Attention**: Reduces computational complexity from O(T²H²W²) to O(T²+H²W²)
- **Self-Subtract Mechanism**: Captures inter-frame temporal inconsistencies  
- **Interpretability**: LRP-based visualization of spatial and temporal attention
- **Robust Performance**: State-of-the-art results on multiple deepfake datasets

## Quick Start

### 1. Installation
```
git clone https://github.com/your-repo/istvt-deepfake
cd istvt-deepfake
pip install -r requirements.txt
```

### 2. Dataset Preparation

Create the following directory structure:
```
data/
├── train/
│ ├── real/
│ └── fake/
├── val/
│ ├── real/
│ └── fake/
└── test/
├── real/
└── fake/
```

Place your video files (.mp4, .avi, .mov) in the appropriate folders.

### 3. Training
```
python train.py
```

Training logs and checkpoints will be saved to `./logs` and `./checkpoints`.

### 4. Inference

Single video:
```
python inference.py --model checkpoints/best_model.pth --video path/to/video.mp4 --visualize
```

Batch processing:
```
python inference.py --model checkpoints/best_model.pth --video_dir path/to/videos/ --output_dir ./results
```

## Model Architecture

ISTVT consists of:

1. **Feature Extractor**: Xception entry flow for texture feature extraction
2. **Decomposed Attention**: Separate spatial and temporal self-attention 
3. **Self-Subtract Mechanism**: Highlights temporal inconsistencies
4. **Transformer Blocks**: 12 layers with 8 attention heads
5. **Classification Head**: Final MLP for binary classification

## Configuration

Modify `config.py` to adjust:
- Model parameters (embed_dim, num_heads, num_layers)
- Training settings (batch_size, learning_rate, num_epochs)
- Data parameters (sequence_length, input_size)

## Key Parameters

- `sequence_length`: Number of frames per video (default: 6)
- `embed_dim`: Feature dimension (default: 728)
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer blocks (default: 12)

## Performance

Achieves state-of-the-art results on:
- FaceForensics++ dataset
- Celeb-DF dataset  
- DFDC dataset
- Cross-dataset generalization

## Note on Paper Reproducibility

This codebase provides all the core functionalities for training, evaluation, and inference of the ISTVT model as described in the paper. However, the following components are **not included**:

- **LRP-based Interpretability/Visualization:** The Layer-wise Relevance Propagation (LRP) method for visualizing spatial and temporal attention, as described in the paper, is not implemented here.
- **Robustness Testing Scripts:** Scripts for evaluating model robustness under JPEG compression, downscaling, and random dropout perturbations are not included.
- **Ablation/Experiment Scripts:** Scripts for running ablation studies on attention variants, sequence lengths, and model depths are not provided.

These components are required for full reproduction of the experimental results and interpretability analyses in the ISTVT paper. The current codebase is fully functional for standard training and inference workflows.

## Citation
```
@article{zhao2023istvt,
title={ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection},
author={Zhao, Cairong and Wang, Chutian and Hu, Guosheng and Chen, Haonan and Liu, Chun and Tang, Jinhui},
journal={IEEE Transactions on Information Forensics and Security},
volume={18},
pages={1335--1348},
year={2023},
publisher={IEEE}
}
```

## License

This project is licensed under the MIT License.
