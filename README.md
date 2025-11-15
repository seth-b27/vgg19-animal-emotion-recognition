# Pet facial expression classification

This notebook implements a transfer-learning pipeline using VGG19 to classify pet facial expressions into four categories: Angry, happy, Other, Sad. It includes dataset preprocessing (resize & split), data augmentation, model training with two batch sizes (32 and 64), and evaluation (confusion matrices, ROC/AUC). While the primary goal is to compare the performance between batch size 32 and 64, , fine-tuning the superior model will also be included.

_**Dataset**_

The dataset should be organized like this:

- `pet_facial_expression_dataset/`
    - `Angry/` (angry pet images)
    - `happy/` (happy pet images)
    - `Other/` (other expressions)
    - `Sad/` (sad pet images)

The preprocessing pipeline automatically splits this into 80% train, 10% validation, and 10% test, then resizes everything to 224×224 for VGG19. A new `preprocess_dataset folder` gets created with the proper train/val/test structure.

Note:  VGG19 requires specific preprocessing that differs from simple normalization. Instead of scaling pixel values to the 0-1 range, VGG19 preprocessing involves mean subtraction based on ImageNet statistics. We handle this with `tf.keras.applications.vgg19.preprocess_input `, and it MUST be applied to train, val, AND test sets identically. 

_**Dataset Pipeline**_

Create TensorFlow dataset pipelines that efficiently load images in batches. Then, we apply data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`, `RandomTranslation`) to make the model more robust before applying vgg19 preprocessing.

_**Model Building**_

After loading pretrained VGG19 base model (`include_top=False`), freeze all the layers so their learned featues stay intact. On top of the frozen base, we add our custom classigication head: 
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(4, activation='softmax')
This head learns to map VGG19's powerful visual features to our four pet emotions. We then compile with Adam optimizer (lr=1e-3), and sparse categorical crossentropy loss.

_**Training**_

Train 2 separate models: one with batch_size=32 and one with batch_size=64, both for 10 epochs with the VGG19 base frozen.

Between experiments, we rebuild the model from scratch so it can be a fair comparison. next, we save the weights from each experiment, and plot the training curves to see which batch size learns faster, more stably, and generalizes better. 

_**Evaluation**_

Time to see how well our models actually work on unseen data

Load the test set (with proper preprocessin') and evaluate both models. Here, we also get to plot ROC curves for each class and calculate AUC scores to measure how well the model can distinguish each emotion.

**-> The batch 64 model consistently outperforms batch 32 across every metric.**

_**Fine Tuning**_

Since batch 64 won the showdown, we take those weights, and fine-tune them by unfreezing the last 4 layers of VGG19. This lets the deeper layers adapt specifically to pet faces while keeping the earlier layers (which learned general visual features) frozen. 

We recompile with a much lower LR (1e-5) to make gentle updates without breaking what's already learned, then train for another 10 epochs. This should typically add another 2-3% accuracy boost as the model's features get more specialized.

_**Key Result**_

**Batch 32**
- train: 87.37% | val: 81% | test: 82%
- decent performance but leaves rroom for improvement
  
**Batch 64**

- train: 93.5% | val: 90% | test: 87%
- +5% improvement on test accuracy, more stable with smoother convergence
  
**Batch 64 Fine-tuned**

- train: 97.62% | val: 93% | test: 89%

_**Usage**_

- Install dependencies: please refer to `requirements.txt` for complete package list. 
- GPU is recommended for training. (CPU will be a lot slower)
