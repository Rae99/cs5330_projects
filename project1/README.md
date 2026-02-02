# Project 1 — Computer Vision Demos

## Overview

- Three small OpenCV apps: `imgDisplay` (image viewer with simple edits), `vid` (webcam pipeline with custom filters, face detection, and depth effects), and `timeBlur` (timing harness for 5x5 blur implementations).
- Custom filters live in include/filters.h and are used across the apps; the main video app also hooks into Depth Anything v2 via ONNX Runtime and a Haar cascade for faces.

## Prerequisites

- CMake 3.16+
- C++20 compiler (tested on clang++ via Apple toolchain)
- OpenCV (brew install opencv)
- ONNX Runtime (brew install onnxruntime)
- Data assets present in `../data` relative to the binaries:
  - `haarcascade_frontalface_alt2.xml` for face detection
  - `model_fp16.onnx` (Depth Anything v2)

## Build

```sh
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Executables

### imgDisplay

- Usage: `./imgDisplay <image>`
- Keys: `q` quit, `o` original, `r` rotate 90° cw, `b` Gaussian blur, `f` flip horizontal, `i` invert colors, `g` grayscale, `d` print image info, `s` save as `output.png`.

### timeBlur

- Usage: `./timeBlur <image>`
- Runs both 5x5 blur variants (`blur5x5_1` and separable `blur5x5_2`) `N=10` times each and prints average seconds per image.

### vid (main webcam app)

- Usage: `./vid` (opens default camera 0). Requires the data files above next to `../data/` relative to the binary.
- Views (mutually exclusive, press again to return to original):
  - `o` original, `g` OpenCV grayscale, `h` custom grayscale, `x` Sobel X, `y` Sobel Y, `m` gradient magnitude, `d` depth map, `D` depth grayscale effect, `e` emboss, `c` face color-pop, `z` depth fog.
- Effects (stackable toggles):
  - `b` custom 5x5 blur, `F` horizontal flip, `v` invert colors, `p` sepia + vignette, `i` blur-quantize posterize, `f` face detect boxes.
- Other controls: `r` rotate 90° cw (accumulates), `s` save current frame to `../output/frame_####.png`, `V` start/stop recording MP4 to `../output`, `t` print frame info, `q` quit.
- Depth controls: depth inference runs every `N=3` frames at scale factor 0.4; adjust in code if you need faster or sharper depth.

## Notes

- Outputs go to `../output` relative to the `vid` binary; create it if missing.
- If the camera opens at a different resolution, recording size follows the capture size.
- DA2 network setup failures print a warning and gracefully skip depth views/effects.
