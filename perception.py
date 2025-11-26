import os
from pathlib import Path
import pandas as pd
from PIL import Image
import random
import numpy as np

class DummyTensor:
    """Represents a torch.Tensor."""
    def __init__(self, shape):
        self.shape = shape
    def __repr__(self):
        return f"Tensor{self.shape}"

class DummyModel:
    """Represents a simple classification model."""
    def __init__(self):
        print("Model: Initializing a dummy ResNet-style classifier for binary classification.")
    def train(self): pass
    def eval(self): pass
    def __call__(self, x):
        # Simulate model output for a batch size of x.shape[0]
        return [DummyTensor((1, 1)) for _ in range(x.shape[0])] # Return list of logits

CURRENT_DIT = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE_DIR = Path(os.path.join(CURRENT_DIT, 'dataset/MURA-v1.1')) 

TRAIN_LABELS_CSV = DATASET_BASE_DIR / "train_labeled_studies.csv"
VALID_LABELS_CSV = DATASET_BASE_DIR / "valid_labeled_studies.csv"

# NEW: Directory where the pre-processed (cleaned) images will be saved
CLEANED_DATA_BASE_DIR = Path(os.path.join(CURRENT_DIT, 'processed_dataset'))
CLEANED_TRAIN_DIR = CLEANED_DATA_BASE_DIR / "train"
CLEANED_VALID_DIR = CLEANED_DATA_BASE_DIR / "valid"

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 5

def get_mura_metadata(labels_csv_path: Path, subset_name: str):
    """
    Loads the MURA dataset metadata (original paths and labels) from CSV.
    Returns a list of (original_image_path, label) tuples.
    """
    print(f"\n--- Loading {subset_name} Metadata via CSV ({labels_csv_path.name}) ---")
    try:
        df = pd.read_csv(labels_csv_path, header=None, names=['StudyPath', 'Label'])
    except Exception as e:
        print(f"ERROR reading CSV {labels_csv_path.name}: {e}")
        return []

    data_tuples = []
    
    for _, row in df.iterrows():
        study_rel_path = Path(row['StudyPath'].strip().strip('/'))
        label = row['Label']
        
        # Construct the absolute path to the study folder
        # We need to go up one level from DATASET_BASE_DIR to match the CSV format
        study_abs_path = DATASET_BASE_DIR.parent / study_rel_path
        
        if study_abs_path.is_dir():
            image_paths = list(study_abs_path.glob('*.png'))
            
            for img_path in image_paths:
                data_tuples.append((str(img_path), label))

    print(f"Total original items (images) loaded for {subset_name}: {len(data_tuples)}")
    return data_tuples


def preprocess_and_save_data(original_data_tuples: list, cleaned_root_dir: Path, subset_name: str):
    """
    Loads original images, applies simple transformation (resize), 
    saves them to the cleaned directory, and returns the new path list.
    """
    print(f"\n--- Preprocessing & Saving {subset_name} Data to {cleaned_root_dir.name} ---")
    
    # Define a simple resize transform
    def resize_transform(image):
        return image.resize(IMAGE_SIZE)

    new_data_tuples = []
    
    # Create the cleaned directory structure
    cleaned_root_dir.mkdir(parents=True, exist_ok=True)
    
    # We will track the structure by the body part (e.g., XR_ELBOW)
    
    for i, (original_path_str, label) in enumerate(original_data_tuples):
        original_path = Path(original_path_str)
        
        parts = original_path.parts
        
        # The body part is the second-to-last directory from the original dataset root (after 'train' or 'valid')
        try:
            split_dir = parts[-5] # 'train' or 'valid'
            body_part = parts[-4] # 'XR_ELBOW'
            patient_study_id = '_'.join(parts[-3:-1]) # 'patient00011_study1_negative'
            image_filename = original_path.name # 'image1.png'
        except IndexError:
            continue
            
        # Define the target subdirectory (e.g., cleaned_dataset/train/XR_ELBOW)
        cleaned_sub_dir = cleaned_root_dir / body_part
        cleaned_sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the final cleaned path
        new_filename = f"{patient_study_id}_{image_filename}"
        cleaned_path = cleaned_sub_dir / new_filename
        
        try:
            image = Image.open(original_path).convert('RGB')
            resized_image = resize_transform(image)
            resized_image.save(cleaned_path)
            
            new_data_tuples.append((str(cleaned_path), label))
        except FileNotFoundError:
            # Skip if the original file is missing
            print(f"WARNING: Original file not found, skipping preprocessing: {original_path.name}")
        except Exception as e:
            print(f"ERROR during processing of {original_path.name}: {e}")
            
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} images and saved to disk...")

    print(f"Finished saving {len(new_data_tuples)} cleaned images to disk.")
    return new_data_tuples



class MURADataset:
    """
    Custom Dataset class for MURA, now reading pre-processed images from disk
    and applying final normalization/tensor conversion.
    """
    def __init__(self, data_tuples, transform=None):
        """
        Args:
            data_tuples (list): List of (preprocessed_image_path, label) pairs.
            transform (callable, optional): Optional transform (for normalization)
        """
        self.data_tuples = data_tuples
        self.transform = transform
        
        if not self.transform:
            self.transform = self._final_transforms()
        
    def __len__(self):
        return len(self.data_tuples)
    
    def __getitem__(self, idx):
        img_path, label = self.data_tuples[idx]
        
        # 1. Read the pre-processed image from the 'cleaned_dataset' directory
        try:
            # Image is already resized, just need to read it
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Cleaned file not found: {img_path}. Skipping.")
            return None 

        # 2. Apply final transforms (Normalization and Tensor conversion)
        image_tensor = self.transform(image)

        # 3. Return the processed image tensor and the label
        label_array = np.array(label, dtype=np.float32)
        
        return image_tensor, label_array
    
    def _final_transforms(self):
        """Applies only normalization and conversion to C x H x W format."""
        def transform_pipeline(image):
            # 1. Convert to Array (The equivalent of 'ToTensor')
            image_array = np.asarray(image, dtype=np.float32) / 255.0 
            # 2. Normalize (Standardize pixel distribution)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_array = (image_array - mean) / std
            # 3. Transpose to C x H x W format (PyTorch requirement)
            image_tensor = np.transpose(image_array, (2, 0, 1))
            
            # The output is a numpy array simulating a PyTorch tensor
            return image_tensor
        return transform_pipeline

def setup_model():
    """
    Sets up a pre-trained CNN model (e.g., ResNet) for binary classification.
    """
    # Placeholder implementation:
    model = DummyModel()
    return model

def train_and_validate(model, train_data_tuples, valid_data_tuples):
    """
    Simulates the main training and validation loop using the loaded data tuples.
    """
    print("\n--- 3. Setting up Data Loaders and Training Loop ---")
    
    # 1. Create Datasets (These now read from the cleaned_dataset directory)
    train_dataset = MURADataset(train_data_tuples)
    valid_dataset = MURADataset(valid_data_tuples)
    
    # 2. Create Data Loaders (This handles batching and shuffling)
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(valid_dataset)}")
    print(f"Simulating DataLoader creation with BATCH_SIZE={BATCH_SIZE}")
    
    # 3. Setup Optimizer and Loss Function (PyTorch placeholders)
    # 4. Training Loop Simulation
    print(f"\n--- Starting Training Simulation for {EPOCHS} Epochs ---")
    
    for epoch in range(EPOCHS):
        # TRAIN STEP SIMULATION
        model.train()
        train_batches = len(train_dataset) // BATCH_SIZE
        print(f"Epoch {epoch+1}/{EPOCHS} (Training): Simulating {train_batches} batches...")
        
        model.eval()
        valid_batches = len(valid_dataset) // BATCH_SIZE
        mock_accuracy = 0.5 + random.random() * 0.4 
        print(f"Epoch {epoch+1}/{EPOCHS} (Validation): Simulating {valid_batches} batches...")
        print(f"Epoch {epoch+1} Complete. Mock Validation Accuracy: {mock_accuracy:.4f}")

    print("\n--- Training Simulation Finished ---")
    print("The model and data are now integrated and ready to run in a PyTorch environment.")


def main():
    """
    Orchestrates the entire MURA classification workflow.
    """
    train_original_tuples = get_mura_metadata(TRAIN_LABELS_CSV, "Training")
    valid_original_tuples = get_mura_metadata(VALID_LABELS_CSV, "Validation")

    train_cleaned_tuples = preprocess_and_save_data(train_original_tuples, CLEANED_TRAIN_DIR, "Training")
    valid_cleaned_tuples = preprocess_and_save_data(valid_original_tuples, CLEANED_VALID_DIR, "Validation")

    model = setup_model()
    
    train_and_validate(model, train_cleaned_tuples, valid_cleaned_tuples)

if __name__ == "__main__":
    main()