from process import ImageKeywording
import argparse


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Deepfake image prediction script.")
    parser.add_argument("--input", required=True, help="Text file with image paths to predict")
    parser.add_argument("--multiple", action="store_true", help="If enabled, processes multiple images from file")
    args = parser.parse_args()
    
    # Lire les chemins du fichier input
    if args.multiple:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        image_paths = [args.input]

    image_keywording = ImageKeywording(image_list=image_paths)

    # Test img /medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg