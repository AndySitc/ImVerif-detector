import os
import pandas as pd

csv_files = {
    "output/alex.csv": "alex",
    "output/andy.csv": "andy",
    "output/abdel.csv": "abdel",
    "output/CoDE.csv": "CoDE",
    "output/freqnet.csv": "amine",
    "output/sahar_trained_on_fr.csv": "sahar_fr",
    "output/sahar_train_on_fs.csv": "sahar_fs",
    "output/sahar_train_on_df40.csv": "sahar_df40"
}

merged_df = None

for file, suffix in csv_files.items():
    if not os.path.exists(file):
        print(f"File {file} not found, skipping.")
        continue

    if "freqnet.csv" in file:
        df = pd.read_csv(file, header=None, names=["image_path", "score", "predicted_label"], on_bad_lines='skip')
    else:
        df = pd.read_csv(file, on_bad_lines='skip')

    # Supprime les lignes parasites où image_path == "image_path" ou NaN
    df = df[df["image_path"].notna() & (df["image_path"] != "image_path")]    

    # print(f"➡️ Colonnes détectées dans {file}: {df.columns.tolist()}")
    # print(df.head())

    df["image_path"] = df["image_path"].astype(str).str.strip()  # Nettoyage

    df = df.rename(columns={
        "score": f"score_{suffix}",
        "predicted_label": f"predicted_label_{suffix}"
    })

    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="image_path", how="outer")

# merged_df["image_path"] = merged_df["image_path"].astype(str)  # Optionnel ici

# Sauvegarde
merged_df.drop_duplicates(inplace=True)
merged_df.to_csv("output/merged_predictions.csv", index=False)
# print("🧪 Nombres d'images uniques :", merged_df["image_path"].nunique())
# print("📄 Nombre total de lignes :", len(merged_df))
# print("🧹 image_path problématiques :")
print(merged_df["image_path"][merged_df["image_path"].str.contains("e-", na=False)])
