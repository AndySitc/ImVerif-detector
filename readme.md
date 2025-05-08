# Installation Guide

## 1. Download Pre-trained Model Weights

Before running predictions, you need to download and set up the model weights.

1. Go to the following link:  
   👉 [Google Drive – Model Weights](https://drive.google.com/drive/folders/1ydUjZHFunWYDgLeZ6jpznfMYFGb7hA9K?usp=drive_link)

2. Download the file `models.zip`.

3. Unzip it.

4. Move the extracted `models` folder into the `ImVerif-detector` directory of the project:


## 2. Set Up the Environment
To install the required virtual environments and dependencies, run:

    bash install_envs_final.sh

## 3. Run predictions
### Predict from a Single Image

    bash run_predictions_final.sh false your/image/path

#### Examble:
    bash run_predictions_final.sh false "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg"

### Predict from a Text File Containing Image Paths

    bash run_predictions_final.sh true path/toyour/file.txt

#### Examble:
    bash run_predictions_final.sh true "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/data/test.txt"






