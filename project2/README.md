# Project 2: Content-Based Image Retrieval (CBIR)

A content-based image retrieval system that supports multiple feature extraction methods (color histograms, texture, deep learning embeddings) and retrieves the top-N matching images from a database. Includes a PyQt6 GUI for interactive querying.

## Author

**Junrui Ding**

## Environment

- **OS:** macOS
- **IDE:** VS Code
- **Build System:** CMake
- **Languages:** C++ (core), Python (GUI extension)

## Time Travel Days

None used.

## Building the Project

```bash
rm -rf build/
mkdir build
cd build
cmake ..
make
```

## Running Executables

### Tasks 1–4: Build and Query Feature Database

```bash
# Step 1: Build feature database
./build_db <image_dir> <output_csv> [task_id]

# Step 2: Query against database
./query_db <target_image> <image_dir> <feature_csv> <topN> [task_id]
```

### Task 5: Deep Learning Embedding Query

```bash
./query_task5 <target_filename> <embedding_csv> <topN>
```

### Task 7: Custom Feature (Grass Detection)

```bash
./query_task7_grass <target_image> <image_dir> <emb_csv> <topN> [--bottom]
```

Use `--bottom` to retrieve the least similar images instead.

## Extension: Interactive GUI

The extension is a PyQt6-based GUI that provides an interactive interface for the CBIR system. Users can select a target image, choose a retrieval method (tasks 1–7), set the number of results, and visually browse the top-N matching images.

### Running the GUI

```bash
python ./query_gui.py
```

> **Note:** Requires Python 3 with PyQt6 installed (`pip install pyqt6`).
