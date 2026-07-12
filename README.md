# Hierarchical-C-XAI
The aim of this project is to design and evaluate a Concept Bottleneck Model (CBM), in which concepts are organized in a hierarchy rather than being treated as an independent flat set. 

The idea behind this work is that semantic concepts naturally exhibit hierarchical relationships. Fine-grained concepts can be combined into high-level semantic concepts, which support the final class prediction.

As a dataset, we took a subset of the ImageNet-1K consisting of 300 images (available in the 'image_subset/' folder). 

<img width="1786" height="591" alt="hierarchical_structure" src="https://github.com/user-attachments/assets/464ea6c4-3f0b-4ab2-95fc-1e803369abd6" />

The dataset was manually reorganized into a semantic hierarchy composed of:

- 4 final classes: Animal, Vehicle, Object, and Plant;
- 11 coarse semantic concepts: representing higher-level semantic groups inferred from fine concepts;
- 10 fine-grained visual concepts: for each coarse concept. They were manually annotated for every image.

The model is compared against a standard ResNet50 baseline.

```python
# Clone the repository
import os
repo_url = "https://github.com/ilariaparodi/Hierarchical-C-XAI.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
%cd Hierarchical-C-XAI
```

## Project Structure

```
├── dataset/
│   ├── dataloaders.py
│   ├── hierarchicalDataset.py
│   ├── split_dataset.py
│   ├── stats.py
│   ├── test.json
│   ├── train.json
│   └── val.json
├── dataset_creation/
│   ├── annotation.py
│   ├── annotations.json
│   ├── concept_vocabulary.json
│   ├── concepts.json
│   └── dataset_extraction.py
├── image_subset/
├── losses.py
├── metrics.py
├── models/
│   ├── resnet_baseline.py
│   └── resnet_hierarchicalCBM.py
├── hierarchical_structure.png
├── results.csv
├── single explanation.py
├── train.py
└── train_hierarchicalCBM.py
```

## File Descriptions
- `dataset_creation/dataset_extraction.py`: extracts the 300-image subset from the original ImageNet-1K dataset.
- `dataset_creation/annotations.json`: contains the annotations of each image.
- `dataset_creation/concepts.json` and `concept_vocabulary.json`: define the concept vocabulary and the hierarchical mappings linking fine concepts to coarse concepts.

- `dataset/hierarchicalDataset.py`: custom PyTorch `Dataset` class to load images and map their corresponding hierarchical targets (fine concepts, coarse concepts and final classes).
- `dataset/dataloaders.py`: configures the PyTorch `DataLoaders` for batching and shuffling train, validation, and test data.
- `dataset/split_dataset.py`: utility script to split the image subset into training, validation, and test sets.
- `dataset/stats.py`: utility script to compute dataset statistics for normalization.
- `dataset/train.json`, `dataset/val.json`, `dataset/test.json`: configuration files containing the data splits with respective image paths and associated annotations.

- `models/resnet_baseline.py`: Standard ResNet50 architecture used as the baseline comparison.
- `models/resnet_hierarchicalCBM.py`: our proposed architecture.
  
- `train.py`: script to train the standard baseline model.
- `train_hierarchicalCBM.py`: script to train the Hierarchical Concept Bottleneck Model (CBM).
- `losses.py`: defines custom loss functions used for training.
- `metrics.py`: contains evaluation metrics to monitor model performance (e.g., final class accuracy, concept precision/recall).
- `results.csv`: contains the final results.
- `single_explanation.py`: inference script to analyze a single image. It outputs the predicted concepts along the hierarchy to visually explain the model's final decision.

## Training
The training scripts automatically save:

- the best model checkpoint
- the training and validation metrics
- the loss plot

When using Google Colab, the results are stored in:

```python
/content/drive/MyDrive/CBM_results/
```

Before running the training, mount your Google Drive with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run the baseline model:

```python
!python train.py
```

Run the Hierarchical CBM:

```python
!python train_hierarchicalCBM.py
```

## Single Explanation
The `single_explanation.py` script allows to select any specific image from the test set and inspect how the hierarchical decisions are made step-by-step. It prints comparison between the Ground-Truth and the Model's Predictions across all 3 levels of the hierarchy (fine concepts, coarse concepts and final class), making the reasoning process of the Hierarchical Concept Bottleneck Model easier to interpret.

To test the script on Google Colab with a custom test image, adjust the IMAGE_PATH string inside the file and run:

```python
!python single_explanation.py
```
