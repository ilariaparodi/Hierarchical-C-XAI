# Hierarchical-C-XAI
The aim of this project is to design and evaluate a Concept Bottleneck Model (CBM), in which concepts are organized in a hierarchy rather than being treated as an independent flat set. 

The idea behind this work is that semantic concepts naturally exhibit hierarchical relationships. Fine-grained concepts can be combined into high-level semantic concepts, which support the final class prediction.

As a dataset, we took a subset of the ImageNet-1K consisting of 300 images (available in the 'image_subset/' folder). 

The dataset was manually reorganized into a semantic hierarchy composed of:

- 4 final classes: Animal, Vehicle, Object, and Plant;
- 11 coarse semantic concepts: representing higher-level semantic groups inferred from fine concepts;
- 10 fine-grained visual concepts: for each coarse concept. They were manually annotated for every image.

The model is compared against a standard ResNet50 baseline.

```bash
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
├── train.py
└── train_hierarchicalCBM.py
```

## Training
The training scripts automatically save:

- the best model checkpoint (`.pt`);
- the training and validation metrics (`.csv`);
- the loss plot (baseline model).

When using Google Colab, the results are stored in:

```
/content/drive/MyDrive/CBM_results/
```

Before running the training, mount your Google Drive with:

```
from google.colab import drive
drive.mount('/content/drive')
```

Run the baseline model:

```
python train.py
```

Run the Hierarchical CBM:

```
!python train_hierarchicalCBM.py
```
