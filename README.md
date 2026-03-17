# Colorizer

> **Work in Progress** — This is a personal research project and is actively being developed.

A U-Net based image colorization model that supports multiple color spaces for comparison.

## Steps to Train Network

1. Setup Repo and Venv

2. Install PyTorch:\
   (For CUDA)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`\
   (For CPU)`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

3. Install dependencies:\
   `pip install -r requirements.txt`

4. Setup Dataset and Weight Bins:\
  `python create_dataset.py`\
  `python create_bin_weights.py`

5. Train Network:\
  `python train.py`

## Demo Colorizer Image

> **add later bash to colorize png image** \
    `download weights from hugging face`
    `python main --...`

## Overview

This project implements an automatic image colorization network that takes grayscale images as input and predicts color. The architecture is based on a U-Net with skip connections, supporting three color space modes:

- **CIELAB** — Standard LAB color space using the 313 bin classification from Zhang et al. (ECCV 2016)
- **OKLab** — Perceptually uniform color space with custom kmeans bins
- **Copic** — Copic marker palette (358 colors), for stylized colorization

## Architecture

- U-Net encoder/decoder with skip connections
- 313/256/358 bin classification output depending on mode
- Softmax + weighted bin decode instead of direct regression

## Credits

See [CREDITS.md](CREDITS.md) for full attribution.

The CIELAB bin lookup table is taken from:

**Colorful Image Colorization**
Richard Zhang, Phillip Isola, Alexei A. Efros — ECCV 2016
https://github.com/richzhang/colorization

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
Third party assets have their own licenses in the `third_party/` folder.