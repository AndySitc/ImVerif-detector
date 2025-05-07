from process import ImageKeywording
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prédiction d'images deepfake.")
    parser.add_argument("--input", required=True, help="Fichier .txt contenant les chemins des images à prédire")
    parser.add_argument("--multiple", action="store_true", help="Si activé, traite plusieurs images listées dans le fichier")
    args = parser.parse_args()
    
    # Lire les chemins du fichier input
    if args.multiple:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        image_paths = [args.input]

    image_keywording = ImageKeywording(image_list=image_paths)

    # Test img /medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg