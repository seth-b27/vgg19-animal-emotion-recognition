# Pet Facial Expression Classification (VGG19 Transfer Learning)

Overview
--------
This notebook implements a transfer-learning pipeline using VGG19 to classify pet facial expressions into four categories: Angry, happy, Other, Sad. It includes dataset preprocessing (resize & split), data augmentation, model training with two batch sizes (32 and 64), and evaluation (confusion matrices, ROC/AUC). While the primary goal is to compare the performance between batch size 32 and 64, , fine-tuning the superior model will also be included.

Project structure
-----------------
- `pet_facial_expression_dataset/` — input raw dataset (expected subfolders: `Angry`, `happy`, `Other`, `Sad`)
- Notebook cells preprocess images and save to:
	- `preprocess_dataset/train/<category>/`
	- `preprocess_dataset/validation/<category>/`
	- `preprocess_dataset/test/<category>/`
- Model weights produced during runs:
	- `weights_batch32.h5`
	- `weights_batch64.h5`
	- `weights_batch64_final-tuned.h5`

Workflow (run cells top-to-bottom)
-------------------------------------------
1. Data preprocessing
- Import required libs (cv2, os, etc.).  
- List and count images in dataset directories.  
- Define `save_images()` and `make_dir()` helpers for resizing and saving.  
- Split dataset into train/validation/test (split ration 80%/10%/10%) 
- Resize all images to 224×224 (VGG19 requirement)
- Save preprocessed images to output directory

Note: VGG19 requires specific preprocessing (mean subtraction, not 0-1 normalization). train, validation, and test sets are applied identically via `tf.keras.applications.vgg19.preprocess_input`

2. Dataset Pipeline Creation

- Create TensorFlow datasets with `image_dataset_from_directory`
- apply data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomTranslation`)
- critical: apply VGG19-specific preprocessing (preprocess_input)

3. Model Building

- Load VGG19 base model (`include_top=False`)
- freeze all VGG19 layers initially
- add custom classification head:

  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(4, activation='softmax')


- Compile with Adam optimizer (lr=1e-3), sparse categorical crossentropy loss

4. Training - Batch Size Comparison

- Experiment 1: Train with batch_size=32 for 10 epochs
- Experiment 2: Train with batch_size=64 for 10 epochs (fresh initialization)
- Save weights after each experiment
- Plot training/validation accuracy and loss curves

5. Evaluation & Analysis

- Load test dataset with proper preprocessing
- Evaluate both models on test set
- Generate confusion matrices for error analysis
- Plot ROC curves and calculate AUC for each class
- Compare performance metrics

6. Fine-Tuning (Optional) the best-performing model weights (batch_size=64)

Key results
-----------
Batch size = 32
- train accuracy = 0.8737
- validation accuracy = 0.8100
- test accuracy = 0.82

Batch size = 64
- train accuracy = 0.9350
- validation accuracy = 0.9000
- test accuracy = 0.87

Batch size = 64 (Fine Tuned - 10 epoch, unfreeze last 4 layers of vgg19)
- train accuracy = 0.9762
- validation accuracy = 0.9300
- test accuracy = 0.89


Usage
-----
1. Prepare dataset following this structure:
	 - `pet_facial_expression_dataset/Angry/`
	 - `pet_facial_expression_dataset/happy/`
	 - `pet_facial_expression_dataset/Other/`
	 - `pet_facial_expression_dataset/Sad/`
	 Place images in the respective folders.

2. Install dependencies: please refer to `requirements.txt` for complete package list. Python 3.11.0 is recommended. 

3. GPU is recommended for training. (CPU will be a lot slower)
