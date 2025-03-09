# Faster R-CNN Implementation in PyTorch

A PyTorch implementation of Faster R-CNN with Feature Pyramid Network (FPN) for object detection. This implementation includes support for ResNet-101 backbone and multi-scale feature detection.

## Features

- ResNet-101 backbone pre-trained on ImageNet
- Feature Pyramid Network (FPN) for multi-scale feature extraction
- Region Proposal Network (RPN) with configurable anchor boxes
- ROI Align for accurate feature pooling
- COCO-style evaluation metrics
- Support for custom datasets
- Comprehensive logging and visualization
- Test-time augmentation
- Gradient accumulation for large batch training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faster-rcnn.git
cd faster-rcnn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
faster-rcnn/
├── src/
│   ├── backbones/
│   │   ├── backbone.py         # VGG16 backbone
│   │   └── resnet_backbone.py  # ResNet101 backbone
│   ├── utils/
│   │   ├── anchor_utils.py     # Anchor box generation and transforms
│   │   ├── config_utils.py     # Configuration loading and setup
│   │   ├── data_utils.py       # Dataset and data loading utilities
│   │   ├── eval_utils.py       # COCO evaluation metrics
│   │   ├── inference_utils.py  # Inference pipeline utilities
│   │   ├── roi_utils.py        # ROI processing utilities
│   │   └── training_utils.py   # Training loop utilities
│   ├── configs/
│   │   └── config_resnet101.yaml  # Model configuration
│   ├── fpn.py                  # Feature Pyramid Network
│   ├── rpn.py                  # Region Proposal Network
│   ├── head.py                 # Detection head
│   ├── train.py               # Training script
│   └── inference.py           # Inference script
```

## Usage

### Training

To train the model:

```bash
python src/train.py --config src/configs/config_resnet101.yaml
```

The training script supports the following arguments:
- `--config`: Path to configuration file (default: config.yaml)
- `--checkpoint`: Path to resume training from checkpoint (optional)

### Inference

To run inference on a directory of images:

```bash
python src/inference.py --config src/configs/config_resnet101.yaml \
                       --checkpoint path/to/model.pth \
                       --input-dir data/images \
                       --output-dir results
```

The inference script supports:
- `--config`: Path to configuration file
- `--checkpoint`: Path to model checkpoint
- `--input-dir`: Directory containing input images
- `--output-dir`: Directory to save visualization results

## Configuration

The model is configured through a YAML file (`config_resnet101.yaml`). Key configuration sections include:

### Dataset Configuration
- Dataset name and target categories
- Category names for visualization
- Cache settings for data loading

### Training Configuration
- Batch size and gradient accumulation steps
- Number of epochs and learning rate
- Optimizer selection (Adam/SGD) and parameters
- Weight decay and momentum settings

### Model Architecture
- Backbone: ResNet-101
- FPN: Feature map channels and levels
- RPN: Anchor box configurations and NMS parameters
- Detection Head: ROI pooling size and detection thresholds

### Image Processing
- Input image size: 1024x1024
- Normalization parameters
- Data augmentation settings

### Evaluation
- COCO AP metrics at multiple IoU thresholds (0.5, 0.75, 0.95)
- Checkpoint saving based on AP@0.5

## Training Process

1. **Data Preparation**:
   - Images are resized to 1024x1024
   - Augmentations include flips, brightness/contrast adjustments
   - Ground truth boxes are mapped to anchor boxes

2. **Feature Extraction**:
   - ResNet-101 backbone extracts features
   - FPN builds multi-scale feature pyramid

3. **Region Proposals**:
   - RPN generates object proposals
   - NMS filters overlapping proposals
   - Proposals are matched to ground truth

4. **Object Detection**:
   - ROI Align extracts proposal features
   - Detection head predicts class and box refinements
   - Training uses balanced sampling of positives/negatives

## Evaluation

The model is evaluated using COCO metrics:
- AP at IoU=0.50:0.95 (primary metric)
- AP at IoU=0.50 (for checkpoint saving)
- AP at IoU=0.75 (strict metric)
- AP at IoU=0.95 (very strict metric)

## License

MIT License

Copyright (c) 2025 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This implementation is based on:
- [Faster R-CNN paper](https://arxiv.org/abs/1506.01497)
- [Feature Pyramid Networks paper](https://arxiv.org/abs/1612.03144)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
