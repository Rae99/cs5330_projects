# CS5330 Project 2 — Content-Based Image Retrieval (CBIR)

**Author:** Junrui Ding  
**Course:** CS5330 Pattern Recognition & Computer Vision  
**Date:** February 2026

This project implements a Content-Based Image Retrieval system using both classical computer vision features and deep learning embeddings. Given a target image, the system finds visually similar images from a database using various feature extraction and distance metric strategies.

---

## Overview

The CBIR system follows a four-step pipeline:

1. **Extract features** from the target image
2. **Extract features** from all database images (can be pre-computed)
3. **Compute distances** between target and database images
4. **Sort and return** the top N most similar matches

This project implements **7 tasks** with different feature types and matching methods, plus a PyQt6 GUI for easy experimentation.

---

## Project Structure

```
project2/
├── src/                    # Source code
│   ├── build_db.cpp        # Feature database builder (Tasks 1-4)
│   ├── query_db.cpp        # Query program (Tasks 1-4)
│   ├── query_task5.cpp     # Deep embedding query (Task 5)
│   ├── query_task7_grass.cpp  # Custom grass detection (Task 7)
│   ├── features.cpp        # Feature extraction implementations
│   ├── ranking.cpp         # Distance metrics
│   ├── csv_io.cpp          # CSV file I/O
│   ├── dir_scan.cpp        # Directory utilities
│   ├── utils.cpp           # Helper functions
│   ├── task_registry.cpp   # Task configuration registry
│   └── query_gui.py        # PyQt6 graphical interface
├── include/                # C++ headers
│   ├── features.h
│   ├── ranking.h
│   ├── csv_io.h
│   ├── dir_scan.h
│   ├── task_registry.h
│   └── utils.h
├── data/                   # Datasets (not in repo)
│   ├── olympus/           # Image database
│   └── ResNet18_olym.csv  # Pre-computed embeddings
├── output/                 # Generated feature CSVs
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
└── .gitignore
```

---

## Tasks Description

### Task 1: Baseline Matching

**Feature:** 7×7 pixel square from the center of the image (RGB values flattened)  
**Distance Metric:** Sum of Squared Differences (SSD)  
**Feature Dimensions:** 147 (7×7×3)

Extracts a 7×7 RGB patch from the image center and compares using Euclidean distance. Simple but effective for images with similar central regions.

**Expected Results (pic.1016.jpg):**
Top matches: pic.0986.jpg, pic.0641.jpg, pic.0547.jpg, pic.1013.jpg

---

### Task 2: Color Histogram Matching

**Feature:** 2D rg chromaticity histogram (whole image)  
**Distance Metric:** Histogram intersection  
**Feature Dimensions:** 256 (16×16 bins)

Uses normalized rg color space (r = R/(R+G+B), g = G/(R+G+B)) to create an illumination-invariant color histogram. Compares histograms using intersection distance: D = 1 - Σ min(H₁, H₂).

**Expected Results (pic.0164.jpg):**
Top matches: pic.0080.jpg, pic.1032.jpg, pic.0461.jpg

---

### Task 3: Multi-Histogram Matching

**Feature:** Two rg chromaticity histograms (whole image + center region)  
**Distance Metric:** Weighted histogram intersection  
**Feature Dimensions:** 512 (2×256)

Combines spatial information by computing separate histograms for the entire image and the central 50%×50% region. Uses weighted combination: D = 0.4×D_whole + 0.6×D_center to emphasize central content.

**Expected Results (pic.0274.jpg):**
Top matches: pic.0273.jpg, pic.1031.jpg, pic.0409.jpg

---

### Task 4: Texture and Color

**Feature:** Color histogram + texture histograms (gradient magnitude + orientation)  
**Distance Metric:** Weighted combination of color and texture distances  
**Feature Dimensions:** 290 (256 color + 16 magnitude + 18 orientation)

Combines color information with texture analysis:

- **Color:** 16×16 rg histogram (256D)
- **Texture:** Sobel gradient magnitudes (16 bins) + orientations (18 bins, 0-180°)
- **Distance:** Equal weighting between color and texture components

Uses Sobel operators to compute gradient magnitude and orientation, capturing edge information complementary to color.

**Expected Results (pic.0535.jpg):**
Shows improved matching compared to Tasks 2-3 by incorporating texture information.

---

### Task 5: Deep Network Embeddings

**Feature:** ResNet18 global average pooling output  
**Distance Metric:** Cosine distance  
**Feature Dimensions:** 512

Uses features extracted from a ResNet18 network pre-trained on ImageNet (1M images, 1000 categories). The 512-dimensional embedding captures high-level semantic content learned from diverse visual patterns.

Cosine distance: D(v₁, v₂) = 1 - cos(θ) = 1 - (v₁·v₂)/(||v₁|| ||v₂||)

**Note:** Features must be pre-computed and provided in CSV format.

**Expected Results:**

- pic.0893.jpg: Finds semantically similar images
- pic.0164.jpg: Compare results with classical features

---

### Task 7: Custom Design (Grass/Lawn Detection)

**Feature:** ResNet18 embeddings + custom green grass features  
**Distance Metric:** Weighted fusion (40% DNN + 60% grass features)  
**Feature Dimensions:** 512 (ResNet18) + 5 (grass)

Custom algorithm for detecting grass/lawn images:

**Grass Features (5D):**

1. Green pixel ratio (HSV thresholding)
2. Average hue in green regions
3. Average saturation in green regions
4. Average value in green regions
5. Has-green flag

**HSV Thresholds:**

- Hue: 35-85 (70°-170° green range)
- Saturation: ≥20
- Value: ≥20
- Morphological operations for noise removal

**Distance Fusion:**

- D_final = 0.4 × D_embedding + 0.6 × D_grass
- Filters out images with <5% green content
- Weighted grass features emphasize hue (×5) and green ratio (×2)

---

## Requirements

### C++ Dependencies

- **CMake** 3.16 or higher
- **C++20** compatible compiler (g++, clang)
- **OpenCV** 4.x with core, imgproc, imgcodecs, highgui modules

### Python Dependencies (GUI only)

- **Python** 3.9+
- **PyQt6** ≥6.4
- **Pillow** (for additional image format support)

Install Python requirements:

```bash
pip install PyQt6 Pillow
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

---

## Build Instructions

### Using CMake (Recommended)

From the project root directory:

```bash
# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Or use make directly
make
```

This produces executables in `build/`:

- `build_db` — Build feature database
- `query_db` — Query with Tasks 1-4
- `query_task5` — Query with Task 5 (deep embeddings)
- `query_task7_grass` — Query with Task 7 (grass detection)

---

## Command-Line Usage

### 1. Build Feature Database (Tasks 1-4)

```bash
./build_db <image_directory> <output_csv> <task_id>
```

**Arguments:**

- `image_directory`: Directory containing image files
- `output_csv`: Output CSV file path
- `task_id`: Task number (1, 2, 3, or 4)

**Example:**

```bash
./build_db ../data/olympus ../output/features_task1.csv 1
```

**Output:** CSV with format: `filename,feature1,feature2,...,featureN`

---

### 2. Query Database (Tasks 1-4)

```bash
./query_db <target_image> <image_dir> <feature_csv> <topN> [task_id]
```

**Arguments:**

- `target_image`: Path to target image
- `image_dir`: Directory containing database images
- `feature_csv`: CSV file generated by build_db
- `topN`: Number of matches to return
- `task_id`: (Optional) Task number, defaults to 1

**Examples:**

```bash
# Task 1: Baseline
./query_db ../data/olympus/pic.1016.jpg ../data/olympus ../output/features_task1.csv 5 1

# Task 2: Color histogram
./query_db ../data/olympus/pic.0164.jpg ../data/olympus ../output/features_task2.csv 5 2

# Task 3: Multi-histogram
./query_db ../data/olympus/pic.0274.jpg ../data/olympus ../output/features_task3.csv 5 3

# Task 4: Color + Texture
./query_db ../data/olympus/pic.0535.jpg ../data/olympus ../output/features_task4.csv 5 4
```

---

### 3. Query with Deep Embeddings (Task 5)

```bash
./query_task5 <target_filename> <embedding_csv> <topN>
```

**Arguments:**

- `target_filename`: Filename only (e.g., `pic.0893.jpg`)
- `embedding_csv`: CSV with ResNet18 embeddings (512D)
- `topN`: Number of matches to return

**Example:**

```bash
./query_task5 pic.0893.jpg ../data/ResNet18_olym.csv 10
```

**Note:** Target must exist in the embedding CSV.

---

### 4. Query with Grass Detection (Task 7)

```bash
./query_task7_grass <target_image> <image_dir> <embedding_csv> <topN>
```

**Arguments:**

- `target_image`: Full path to target image
- `image_dir`: Directory containing database images
- `embedding_csv`: CSV with ResNet18 embeddings
- `topN`: Number of matches to return

**Example:**

```bash
./query_task7_grass ../data/olympus/pic.0408.jpg ../data/olympus ../data/ResNet18_olym.csv 10
```

**Best Results:** Use images with visible grass/lawns as targets.

---

## GUI Usage

The PyQt6 graphical interface provides a user-friendly way to run queries across all tasks.

### Launch GUI

From the project root:

```bash
python src/query_gui.py
```

### GUI Features

#### Intelligent Target Selection

- **Dropdown:** Browse all images
- **Search:** Type to filter (e.g., "pic.1016")
- **Partial Input:** Type "0345" to match "pic.0345.jpg"
- **Smart Matching:** Handles partial extensions ("pic.0022.j" → "pic.0022.jpg")
- **Keyboard Navigation:** Use ◀/▶ buttons or P/N keys

#### Automatic CSV Management

- **Tasks 1-4:** Click "Build" to auto-generate features
- **Tasks 5/7:** Click "Browse" to select pre-computed embeddings
- **Smart Caching:** Remembers CSV for each task (persists across sessions)
- **Auto-Validation:** Checks CSV dimensions match task requirements

#### Collapsible Interface

- **Controls Panel:** Auto-hides after running query (click to toggle)
- **Program Output:** Hidden by default, click to view debug info
- **Keyboard Shortcuts:**
  - `R` — Run query
  - `P/N` — Previous/Next target
  - `←/→` — Page navigation
  - `Q/ESC` — Quit

#### CSV Persistence

The GUI uses Qt's QSettings to remember CSV paths for each task. On macOS, settings are stored in:

```
~/Library/Preferences/CS5330/CBIR.plist
```

This means:

- Switch from Task 1 → Task 2: CSV auto-updates to features_task2.csv
- Close and reopen: All CSV associations are restored
- No need to re-select CSVs when switching tasks

---

## Implementation Details

### Feature Extraction (features.cpp)

#### Task 1: Baseline

```cpp
bool compute_task1_feature(const cv::Mat &img, std::vector<float> &feat);
```

Extracts 7×7×3 = 147 RGB values from image center.

#### Task 2: Color Histogram

```cpp
bool compute_task2_feature(const cv::Mat &img, std::vector<float> &feat);
```

Computes 16×16 rg chromaticity 2D histogram, normalized to sum=1.

#### Task 3: Multi-Histogram

```cpp
bool compute_task3_feature(const cv::Mat &img, std::vector<float> &feat);
```

Concatenates whole-image and center-region rg histograms (512D total).

#### Task 4: Color + Texture

```cpp
bool compute_task4_feature(const cv::Mat &img, std::vector<float> &feat);
```

Combines:

- Color: 16×16 rg histogram (256D)
- Texture: Sobel gradient magnitude histogram (16 bins)
- Texture: Sobel gradient orientation histogram (18 bins, unsigned 0-180°)

#### Task 7: Grass Features

```cpp
bool extract_grass_features(const cv::Mat &img, std::vector<float> &feat);
```

Extracts 5D grass-specific features using HSV color space and morphological operations.

### Distance Metrics (ranking.cpp)

#### SSD Distance (Task 1)

```cpp
float ssd_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Sum of squared differences: D = Σ(aᵢ - bᵢ)²

#### Histogram Intersection (Tasks 2-4)

```cpp
float hist_intersection_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Intersection distance: D = 1 - Σ min(aᵢ, bᵢ)

#### Multi-Histogram Distance (Task 3)

```cpp
float task3_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Weighted combination of two histogram intersections with 40% whole, 60% center weighting.

#### Color+Texture Distance (Task 4)

```cpp
float task4_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Equal weighting between color distance and texture distance (average of magnitude and orientation).

#### Cosine Distance (Task 5)

```cpp
float cosine_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Cosine distance: D = 1 - (v₁·v₂)/(||v₁|| ||v₂||)

#### Grass Distance (Task 7)

```cpp
float grass_distance(const std::vector<float> &a, const std::vector<float> &b);
```

Weighted Euclidean distance with emphasis on green ratio (×2) and hue (×5).

---

## Data Files

### Image Database

Place image dataset in `data/olympus/`. Supports formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.ppm`

### ResNet18 Embeddings

The `data/ResNet18_olym.csv` file contains 512-dimensional feature vectors extracted from a ResNet18 network pre-trained on ImageNet. Format:

```
filename,f1,f2,...,f512
pic.0001.jpg,0.123,0.456,...,0.789
```

**Note:** This file is required for Tasks 5 and 7 but is not included in the repository due to size. It can be generated using a pre-trained ResNet18 model.

---

## Build Instructions

### Prerequisites

- CMake 3.16 or higher
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- OpenCV 4.x

### Compilation

```bash
# From project root
mkdir -p build
cd build
cmake ..
cmake --build .

# Verify executables were created
ls -la build_db query_db query_task5 query_task7_grass
```

**Common Issues:**

- If OpenCV is not found, specify path: `cmake -DOpenCV_DIR=/path/to/opencv ..`
- On macOS with Homebrew: `brew install opencv`
- On Ubuntu: `sudo apt-get install libopencv-dev`

---

## Detailed Usage Examples

### Example 1: Complete Workflow for Task 1

```bash
# Step 1: Build feature database
./build/build_db data/olympus output/features_task1.csv 1

# Step 2: Query with target image
./build/query_db data/olympus/pic.1016.jpg data/olympus output/features_task1.csv 5 1

# Expected output:
# Top 5 matches for target: data/olympus/pic.1016.jpg
# 1) pic.0986.jpg  dist=14049  fullpath=data/olympus/pic.0986.jpg
# 2) pic.0641.jpg  dist=21756  fullpath=data/olympus/pic.0641.jpg
# ...
```

### Example 2: Task 5 with Deep Embeddings

```bash
# No build step needed - embeddings are pre-computed
./build/query_task5 pic.0893.jpg data/ResNet18_olym.csv 10

# Expected output:
# Top 10 matches (Task5 cosine) for target: pic.0893.jpg
# 1) pic.0897.jpg  dist=0.15177
# 2) pic.0136.jpg  dist=0.17616
# ...
```

### Example 3: Task 7 Grass Detection

```bash
./build/query_task7_grass data/olympus/pic.0408.jpg data/olympus data/ResNet18_olym.csv 10

# Expected output:
# Target green ratio: 0.465387
# Task 7: Grass/Lawn Detection - Top 10 matches
# 1. pic.0856.jpg (distance: 0.181994)
# 2. pic.1098.jpg (distance: 0.182133)
# ...
```

---

## GUI Detailed Guide

### First-Time Setup

1. **Launch GUI:**

   ```bash
   python src/query_gui.py
   ```

2. **Select Image Directory:**
   - Click "Browse" next to "Image Dir"
   - Navigate to `data/olympus`
   - GUI loads all images and populates target dropdown

3. **For Tasks 1-4 (Build Features):**
   - Select desired task from "Task" dropdown
   - Click "Build" button
   - Wait for progress dialog
   - CSV is auto-generated and validated

4. **For Tasks 5/7 (Use Pre-computed):**
   - Select Task 5 or 7
   - Click "Browse" next to CSV
   - Select `data/ResNet18_olym.csv`
   - Validation confirms 512 features

5. **Run Query:**
   - Select or type target image (e.g., "pic.1016")
   - Set Top K value
   - Click "▶ Run Query (R)"
   - Controls auto-hide, results display

### GUI Workflow Scenarios

#### Scenario 1: Testing Multiple Targets (Same Task)

```
1. Task 1, target: pic.1016 → Run → Results shown
2. Change target: pic.0435 → Run → New results (uses cached CSV)
3. Change target: pic.0164 → Run → New results (still uses same CSV)
No need to re-select CSV!
```

#### Scenario 2: Comparing Different Tasks

```
1. Task 1 → features_task1.csv (cached) → pic.1016 → Run
2. Switch to Task 2 → CSV auto-switches to features_task2.csv → Run
3. Switch to Task 5 → CSV auto-switches to ResNet18_olym.csv → Run
All CSVs remembered!
```

#### Scenario 3: Quick Target Input

```
Type in target box: "0345" → Press Enter
→ Auto-matches to "pic.0345.jpg"
→ Click Run → Query executes

Or: Type "pic.0022.j" → Press Enter
→ Auto-completes to "pic.0022.jpg"
→ Click Run → Query executes
```

---

## GUI Technical Features

### CSV Validation

The GUI automatically validates CSV files against expected feature dimensions:

| Task | Expected Dimensions | Feature Type            |
| ---- | ------------------- | ----------------------- |
| 1    | 147                 | 7×7×3 pixel square      |
| 2    | 256                 | 16×16 rg histogram      |
| 3    | 512                 | 2×256 multi-histogram   |
| 4    | 290                 | 256+16+18 color+texture |
| 5    | 512                 | ResNet18 embedding      |
| 7    | 512                 | ResNet18 embedding      |

If dimensions don't match, a warning is displayed with guidance on which CSV to use.

### Target Matching Strategies

The GUI implements a 5-tier intelligent matching system:

1. **Exact Match:** Input exactly matches a filename
2. **Case-Insensitive:** Matches ignoring case
3. **Stem Match:** Removes partial extensions (`.j`, `.jp`) and matches base name
4. **Prefix Match:** Finds first filename starting with input
5. **Substring Match:** Finds first filename containing input

If no match is found when clicking Run, a warning dialog appears.

### Session Persistence

The GUI uses Qt's QSettings to persist state:

- **Stored:** CSV path for each task (1-7)
- **Location (macOS):** `~/Library/Preferences/CS5330/CBIR.plist`
- **Location (Linux):** `~/.config/CS5330/CBIR.conf`
- **Behavior:** Next launch automatically restores all CSV associations

---

## Troubleshooting

### Build Issues

**Problem:** `CMake Error: Could not find OpenCV`

```bash
# Solution 1: Install OpenCV
brew install opencv              # macOS
sudo apt install libopencv-dev   # Ubuntu

# Solution 2: Specify OpenCV path
cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 ..
```

**Problem:** `undefined reference to cv::...`

```bash
# Ensure CMakeLists.txt links OpenCV properly:
target_link_libraries(query_db ${OpenCV_LIBS})
```

### Runtime Issues

**Problem:** "Executable not found"

- **Cause:** Build folder not detected by GUI
- **Solution:** Ensure build/ exists in project root or set BUILD_CANDIDATES in query_gui.py

**Problem:** "CSV dimension mismatch"

- **Cause:** Using wrong CSV for selected task
- **Solution:** Check CSV validation message; rebuild CSV for correct task or select appropriate CSV

**Problem:** Images not displaying in GUI

- **Cause:** Fullpath in CSV incorrect or image files moved
- **Solution:** Enable "Show Program Output" and check file paths; ensure image_dir matches CSV generation directory

**Problem:** "Target filename not found in embedding csv"

- **Cause:** Task 5 target not in ResNet18_olym.csv
- **Solution:** Choose a different target that exists in the embedding CSV

### GUI Issues

**Problem:** GUI crashes on file selection

- **Cause:** Qt/PyQt6 version compatibility
- **Solution:** Ensure PyQt6 ≥6.4, or downgrade to PyQt6==6.6.1

**Problem:** CSV not remembered after restart

- **Cause:** Settings file corrupted or permissions
- **Solution:** Delete `~/Library/Preferences/CS5330/CBIR.plist` and restart

---

## Testing

### Verification Tests

Run these tests to verify correct implementation:

**Task 1 (Baseline):**

```bash
./build/query_db data/olympus/pic.1016.jpg data/olympus output/features_task1.csv 4 1
```

Expected top 4: pic.0986.jpg, pic.0641.jpg, pic.0547.jpg, pic.1013.jpg

**Task 2 (Histogram):**

```bash
./build/query_db data/olympus/pic.0164.jpg data/olympus output/features_task2.csv 3 2
```

Expected matches should be blue/architectural images

**Task 5 (Deep Network):**

```bash
./build/query_task5 pic.0893.jpg data/ResNet18_olym.csv 5
```

Expected: Semantically similar images (fire hydrants, outdoor objects)

**Task 7 (Grass):**

```bash
./build/query_task7_grass data/olympus/pic.0408.jpg data/olympus data/ResNet18_olym.csv 10
```

Expected: Images with grass/lawns; green ratio printed for target

---

## Project Report

The project report (report.pdf) includes:

1. Short description of the overall project
2. Required result images for each task
3. Description and examples of extensions
4. Reflection on learning outcomes
5. Acknowledgements and references

---

## Extensions Implemented

### 1. PyQt6 Graphical User Interface

- Comprehensive GUI with image thumbnails
- Intelligent target selection with autocomplete
- Automatic CSV management and caching
- Collapsible panels for cleaner interface
- Session persistence with QSettings

### 2. Grass/Lawn Detection (Task 7)

- Custom HSV-based green detection
- Morphological filtering for robust segmentation
- Fusion of deep embeddings with domain-specific features
- Weighted distance metric optimized for grass detection

### 3. Task Registry System

- Modular design for easy task addition
- Centralized configuration in task_registry.cpp
- Lambda support for custom distance functions

---

## Development Notes

### Code Organization

- **Separation of Concerns:** Feature extraction, distance metrics, and I/O are in separate modules
- **Task Registry Pattern:** New tasks can be added by registering in task_registry.cpp
- **CSV Format:** Simple comma-separated format for easy debugging and portability

### Design Decisions

- **Multi-program approach:** Separate build and query steps for efficiency
- **CSV storage:** Text format for transparency and cross-platform compatibility
- **GUI independence:** Core C++ code independent of Python GUI

---

## Performance Considerations

- **Database Size:** With ~1000 images, query time is typically <1 second
- **Build Time:** Feature extraction for Task 1-4 takes ~10-60 seconds depending on image count
- **Memory:** Entire feature database is loaded into memory for fast querying

---

## Known Limitations

1. **Task 5/7 Embeddings:** Require pre-computed CSV (not generated by this project)
2. **Image Formats:** Limited to common formats supported by OpenCV
3. **Feature CSV Size:** Can be large for high-dimensional features (Task 3: 512×N floats)

---

## Future Enhancements

- [ ] Real-time feature extraction (no CSV pre-computation)
- [ ] Support for custom neural network embedding extraction
- [ ] Additional texture features (Gabor filters, co-occurrence matrices)
- [ ] Query by region of interest (ROI-based retrieval)
- [ ] Batch query mode for multiple targets

---

## Acknowledgements

- **Course:** CS5330 Pattern Recognition & Computer Vision, Northeastern University
- **Instructor:** Professor Bruce Maxwell
- **References:**
  - Shapiro & Stockman, _Computer Vision_, Chapter 8
  - ResNet18 architecture from torchvision/PyTorch
  - OpenCV documentation for image processing functions

---

## License

This project is for educational purposes only (CS5330 coursework).

---

## Contact

For questions or issues, please contact through the course Canvas page.
