import pandas as pd

# Charger le CSV
csv_path = "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/data/test_dataset_balanced.csv"
df = pd.read_csv(csv_path)

# Extraire la colonne des chemins d'image
image_paths = df['frame_path']

# Sauvegarder dans un fichier .txt
output_txt = "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/data/test_dataset_balanced.txt"
image_paths.to_csv(output_txt, index=False, header=False)

print(f"Les chemins ont été enregistrés dans : {output_txt}")
