# Experiments

> Running log of experiments, changes, and observations for the colorizer project.
> Dataset: MIRFLICKR-25K

---

## Weight Initialization

### Kaiming Initialization
- **Status:** Abandoned
- **Notes:** Tested Kaiming (He) initialization. Did not yield better results for this architecture.

### Normal Initialization
- **Status:** Current
- **Notes:** Switched to standard normal initialization. More stable training behavior.

---

## Discriminator Architecture

### Single Output Discriminator
- **Status:** Abandoned
- **Notes:** Standard single scalar real/fake output. Less spatial feedback for the generator.

### PatchGAN Discriminator
- **Status:** Current
- **Notes:** Switched to PatchGAN which classifies overlapping patches of the image as real or fake rather than the whole image. Provides more spatially detailed gradient signal to the generator, improving local color consistency.

---

## Output Method

### Direct ab Regression
- **Status:** Abandoned
- **Notes:** Network directly predicted 2 ab channel values per pixel. Produced brownish/desaturated output due to the averaging problem — the network hedges toward the mean color when uncertain.

### Tanh + Scaling (*128)
- **Status:** Abandoned
- **Notes:** Added Tanh activation to final layer and scaled output by 128. Made results worse, likely due to gradient squashing near Tanh boundaries destabilizing training.

### 313 Bin Classification (CIELAB)
- **Status:** Current
- **Notes:** Switched final output to 313 bin classification following Zhang et al. (ECCV 2016). Softmax over 313 bins followed by weighted sum decode using pts_in_hull lookup table. Resolved brownish tint. Output is now close to ECCV baseline but still shows wrong colors in places and some blurring.

---

## Network Size

### Smaller Network
- **Status:** Abandoned
- **Notes:** Earlier smaller feature size. Insufficient capacity for colorization task.

### Larger Network
- **Status:** Current
- **Notes:** Increased feature size. Improved output quality.

---

## Dilated Convolutions

### No Dilation
- **Status:** Abandoned
- **Notes:** All encoder layers used standard convolutions with limited receptive field at the bottleneck.

### Dilation on conv5 (dilation=2)
- **Status:** Current
- **Notes:** Added dilation=2 to the deepest encoder layer (conv5) with stride=1 and padding=2 to preserve spatial size at the bottleneck (14x14). Expands receptive field without changing spatial dimensions, allowing the network to capture more global context for color decisions. Resulted in a minor improvement in output quality. No size mismatches — architecture works correctly as is.

---

## Planned Experiments

- [ ] OKLab 256 bin classification with custom kmeans bins
- [ ] Copic 358 bin classification for stylized colorization
- [ ] Rebalanced cross entropy loss (Zhang et al. siggraph17) to upweight rare colors
- [ ] Compare CIELAB vs OKLab vs Copic output quality side by side
- [ ] Additional dilation layers (conv4 + conv5) to further expand receptive field
- [ ] Larger training dataset beyond MIRFLICKR-25K
