# Dataset Organization

## Overview

This dataset is released in four companion parts, all with predefined train, val, and test splits: 

1. a **classification image dataset** organized by decomposition stage
2. a **segmentation dataset** containing full-body images and corresponding binary masks
3. an **image-to-text dataset** provided in JSON format and
4. a **multimodal metadata dataset** for classification provided in CSV format.

### Classification Image Dataset

The classification dataset is organized by split and decomposition stage.

```text
PMI_Class_Split/
├── train/
├── val/
└── test/
    ├── FRESH/images/
    ├── BLOAT/images/
    ├── ACTIVE_DECAY/images/
    └── ADVANCED_DECAY/images/

```
---
### Segmentation Dataset
The segmentation dataset is organized by split and full body binary masks
```text
Segmentation/
├── train/
├── val/
└── test/
    ├──Images/
    ├── Masks
```

### Image-to-Text Caption
```text
caption_splits/
├── train_captions.json
├── val_captions.json
└── test_captions.json
```
### Multimodal Metadata Dataset

```text
metadata_split/
├── train_metaData.csv
├── val_metaData.csv
└── test_metaData.csv
```

### Dataset loading examples are given in this repository.
