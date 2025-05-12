# ImVerif-detector installation Guide

## 1. Download Pre-trained Model Weights

Before running predictions, you must download the pre-trained model weights.

Please ensure that at least **45 GB** of free disk space is available before proceeding with the environment setup and model installation.

To download and install the model weights, simply run the following command from the root of the repository:

    bash install_models.sh

This script will automatically:
- Download the models.zip archive from the designated remote storage,
- Unzip its contents,
- And place the resulting models folder inside the ImVerif-detector project directory.

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





