import os

# Dossier contenant les images
image_directory = "id0_0000/"  # Remplace par le chemin de ton dossier d'images
output_file = "test.txt"  # Nom du fichier de sortie

# Vérifier si le dossier existe
if not os.path.isdir(image_directory):
    print("Le dossier spécifié n'existe pas.")
else:
    # Ouvrir le fichier en mode écriture
    with open(output_file, "w") as f:
        # Parcourir le dossier et ses sous-dossiers
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                # Vérifier si l'extension du fichier est une image
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Chemin absolu de l'image
                    file_path = os.path.join(root, file)
                    f.write(file_path + "\n")
    
    print(f"Les chemins des images ont été enregistrés dans {output_file}.")
