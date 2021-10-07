# GAN-image-detection

## Prerequisite
1. Create the conda environment
```bash
conda env create -f environment.yml
```
2. Download the model's weights from [here](https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip?dl=0) and unzip it under the main folder
```bash
wget https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip?dl=0
unzip weigths.zip
```

## Test the detector on a single image
We provide a simple script to obtain the model score for a single image.
```bash
python gan_vs_real_detector.py --img_path $PATH_TO_TEST_IMAGE
```

## Performances
We provide a [notebook](https://github.com/polimi-ispl/GAN-image-detection/roc_curves.ipynb) with the script for computing the ROC curve for each dataset.
