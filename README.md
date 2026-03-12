# Colorizer

> **Work in Progress** — This is a personal research project and is actively being developed.

A U-Net based image colorization model that supports multiple color spaces for comparison.

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