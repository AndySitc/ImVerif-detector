# ImVerif-detector installation Guide

## 1. Download Pre-trained Model Weights
Ensure that at least 45 GB of free disk space is available before proceeding the intallation of the environments + the download of the models.

Before running predictions, you need to download and set up the model weights.

1. Go to the following link:  
   👉 [Google Drive – Model Weights](https://drive.google.com/drive/folders/1ydUjZHFunWYDgLeZ6jpznfMYFGb7hA9K?usp=drive_link)

2. Download the file `models.zip`.

3. Unzip it.

4. Move the extracted `models` folder into the `ImVerif-detector` directory of the project:


## 2. Set Up the Environment
To install the required virtual environments and dependencies, run:

    bash install_envs_test.sh /your/custom/env/base/path

### Example
    bash install_envs_test.sh /medias/db/ImagingSecurity_misc/sitcharn

## 3. Run predictions
### Predict from a Single Image

    bash run_predictions_test.sh false /path/to/your/image.jpg /your/custom/env/base/path

#### Examble:
    bash run_predictions_test.sh false "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg" /medias/db/ImagingSecurity_misc/sitcharn


### Predict from a Text File Containing Image Paths

    bash run_predictions_test.sh true /path/to/your/file.txt /your/custom/env/base/path

#### Examble:
    bash run_predictions_test.sh true "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/data/test.txt" /medias/db/ImagingSecurity_misc/sitcharn


You will find the final prediction results in: output/ensemble.csv





