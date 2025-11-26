# MURA Dataset — End-to-End Data Processing Pipeline

### Dataset Link: https://stanfordaimi.azurewebsites.net/datasets/3e00d84b-d86e-4fed-b2a4-bfe3effd661b

A concise, well-documented pipeline to extract metadata, preprocess (resize) MURA images once, persist cleaned images on disk, and provide a lightweight dataset loader that performs final normalization at training time.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Layout](#project-layout)
- [Processing Stages](#processing-stages)
  - [Stage 1 — Metadata Extraction & Path Resolution](#stage-1---metadata-extraction--path-resolution)
  - [Stage 2 — Data Cleaning & Persistence](#stage-2---data-cleaning--persistence)
  - [Stage 3 — On-the-Fly Loading & Final Standardization](#stage-3---on-the-fly-loading--final-standardization)
- [API / Key Functions](#api--key-functions)
- [Configuration (constants)](#configuration-constants)
- [Notes](#notes)
- [License](#license)

## Overview
This repository implements a three-stage data pipeline tailored for the MURA musculoskeletal radiographs dataset. The heavy image transform (resizing) is executed once and stored under `processed_dataset/`. During training the loader only performs normalization and array/tensor conversion.

## Quick Start

1. Place the unmodified MURA dataset under:
   - Per this code, expected base: `dataset/MURA-v1.1/`
2. Install dependencies:
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
   - If a requirements file is not present, install common libs:
     ```bash
     pip install pandas pillow numpy
     ```
3. Run the pipeline:
   ```bash
   python3 perception.py
   ```

## Prerequisites

- Python 3.8 or later
- Pip (Python package installer)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yasirKhaan/mura_dataset_pipeline.git
   cd mura_dataset_pipeline
   ```
2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- To run the entire pipeline (metadata extraction, data cleaning, and normalization):
  ```bash
  python3 perception.py
  ```
- To only preprocess and save the data (Stage 2):
  ```bash
  python3 perception.py --preprocess_only
  ```

## Project Layout

```
mura_dataset_pipeline/
│
├── dataset/                  # Place the unmodified MURA dataset here
│   └── MURA-v1.1/
│       ├── train/
│       ├── valid/
│       └── ...
│
├── processed_dataset/        # Resized and processed images will be saved here
│   ├── train/
│   └── valid/
│
├── mura_classifier.py        # Main script for the data processing pipeline
├── train.py                  # Training script (to be provided)
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## Processing Stages

### Stage 1 — Metadata Extraction & Path Resolution

The MURA dataset uses a deep directory structure where the final binary label (Positive/Negative) is not easily extractable from the folders themselves. The ground truth labels are provided in separate CSV files.

**Techniques Performed:**

- **CSV Parsing (Pandas):** The `get_mura_metadata` function uses the Pandas library to read the `train_labeled_studies.csv` and `valid_labeled_studies.csv` files.

- **Path Reconstruction:** The CSV files provide relative study paths (e.g., `MURA-v1.1/train/XR_SHOULDER/patient00001/study1_positive/`). This relative path must be joined with the system's base directory (`DATASET_BASE_DIR`) to find the actual study folder on disk.

- **Image Path Mapping:** For every study path listed in the CSV, the script recursively finds all `.png` images within that study folder using `Path.glob()`. This creates the initial dataset structure: a list of tuples `(original_image_path, binary_label)`.

**Output of Stage 1:** A list of all original image file paths, correctly paired with their binary classification label (0 or 1).

### Stage 2 — Data Cleaning & Persistence

This stage performs the essential image transformation that needs to be done only once, saving the results to the new `cleaned_dataset` directory.

**Techniques Performed:**

- **Target Directory Creation:** The function `preprocess_and_save_data` first ensures that the new directories (`cleaned_dataset/train`, `cleaned_dataset/valid`) exist.

- **Image Loading (PIL):** For each image in the list from Stage 1, the image is loaded from the original path using the Python Imaging Library (`PIL.Image.open`).

- **Preprocessing Transformation (Resizing):**

  - The loaded image is converted to RGB format (`.convert('RGB')`) to ensure channel consistency, even if the source is grayscale.

  - The image is then resized to a fixed dimension (`IMAGE_SIZE = 224x224`). This step standardizes the input dimensions, which is mandatory for feeding images into pre-trained Convolutional Neural Networks (CNNs).

- **File Saving:** The resized image is saved to the corresponding sub-directory within `cleaned_dataset`. The new file path is constructed to flatten the structure slightly while retaining crucial information (e.g., `XR_ELBOW/patient00011_study1_negative_image1.png`).

**Output of Stage 2:** The `cleaned_dataset` directory on disk containing all resized images, and an updated list of tuples `(cleaned_image_path, binary_label)`.

### Stage 3 — On-the-Fly Loading & Final Standardization

In this final stage, the `MURADataset` class and the subsequent training pipeline utilize the pre-cleaned data.

**Techniques Performed:**

- **Dataset Implementation:** The custom `MURADataset` class is defined to interact with the PyTorch (or TensorFlow) training loop. It reads the pre-processed images from the `cleaned_dataset` directory.

- **Final Normalization:** After the image is loaded, a final standardization is applied:

  - **Conversion to NumPy/Tensor:** The image pixels are converted to a floating-point NumPy array and scaled from $0-255$ to $0.0-1.0$ (min-max normalization).

  - **Z-Score Normalization:** The most critical step is standardizing the pixel values using the mean and standard deviation derived from the large ImageNet corpus. This formula ensures the data distribution matches the expected input of pre-trained models:

    $$I_{norm} = \frac{I_{scaled} - \mu}{\sigma}$$

    Where $\mu$ and $\sigma$ are the ImageNet mean and standard deviation for the R, G, and B channels.

  - **Dimension Transposition:** The image dimensions are transposed from the standard PIL format (Height x Width x Channels) to the PyTorch/CNN required format (Channels x Height x Width).

- **Integration:** The final normalized array and the label are returned, ready to be grouped into batches by the DataLoader and fed directly to the deep learning model.

This three-stage process ensures that the most time-consuming step (resizing/cleaning) is performed offline, while the final, necessary normalization is performed efficiently during the training run.

## Key Functions

- `get_mura_metadata(csv_file: str) -> List[Tuple[str, int]]`: Extracts metadata from the given CSV file and returns a list of tuples containing image file paths and binary labels.

- `preprocess_and_save_data(image_path: str, target_path: str) -> None`: Loads an image from the given path, preprocesses (resizes) it, and saves it to the target path.

- `MURADataset`: A PyTorch Dataset class that loads pre-processed images from the `cleaned_dataset` directory and applies final normalization.

## Configuration (constants)

- `DATASET_BASE_DIR`: The base directory where the MURA dataset is located.
- `IMAGE_SIZE`: The target size (width, height) for resizing images. Default is `(224, 224)`.
- `BATCH_SIZE`: The number of samples per batch during training. Default is `32`.
- `NUM_WORKERS`: The number of subprocesses to use for data loading. Default is `4`.

## Notes

- Ensure that the directory structure and file permissions allow for reading the source images and writing to the target directories.
- It is recommended to use a virtual environment to manage Python dependencies for this project.
- The training script (`train.py`) is not provided in this repository. This is a placeholder for integrating the actual model training code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
