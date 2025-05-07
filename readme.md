## Installation guide

To install all the virtual environement and all the libraries run the 2 following line in your terminal: 

    chmod +x install_envs.sh
    ./install_envs.sh

Test to get the prediction for each model (demo for me):

    Sahar's model
        python demo.py -f examples/fake.png -m weights/model_epoch_best.pth

    Andy's model
        python model_to_excel.py

    Abdel's model
        python "test_deepfake_detector copy.py"

## To get the prediction
### Image path of test (fake)

    /medias/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg


### Get the prediction:
    chmod +x run_predictions.sh 
    ./run_predictions.sh /chemin/vers/ton/image.jpg

#### MAJ get prediction
    bash run_predictions_test.sh true paths.txt  # for multiple predictions
    bash run_prediction.sh false /chemin/image.jpg  # pour une seule


#### Examble:
    ./run_predictions.sh "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg"  
    
    bash run_prediction.sh false "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg"  
