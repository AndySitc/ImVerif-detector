import argparse
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



def print_log(txt, log_file=None):
    print(txt)
    if log_file:
        log_file.write(f"{txt}\n")

print_log("Starting the script...")

# ------------------ Dataset Class ------------------
class DetectorsOutputDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ------------------ Meta Classifier ------------------
class MetaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MetaClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.classifier(x)

# ------------------ Info Printer ------------------
def print_data_info(df, features, class_cols, args, mode="train"):
    print_log("============= Dataset Information =============", log)
    print_log(f"Mode                  : {mode.upper()}", log)
    print_log(f"CSV Path              : {args.csv_path if mode == 'train' else args.test_csv}", log)
    print_log(f"Data normalization    : {args.data_norm}", log)
    print_log(f"Using CLIP Embeddings : {args.use_clip}", log)
    if args.use_clip:
        clip_path = args.train_clip_csv_path if mode == 'train' else args.test_clip_csv_path
        print_log(f"CLIP CSV Path         : {clip_path}", log)
        print_log(f"Top-k CLIP Embeddings : {args.top_k_clip}", log)
    print_log(f"Total Samples         : {len(df)}", log)
    print_log(f"Feature Dimension     : {features.shape[1]}", log)
    print_log(f"Detector Columns      : {[cls_col.split('_')[-1] for cls_col in class_cols]}", log)
    print_log(f"Number of Detectors   : {len(class_cols)}", log)
    class_counts = df["correct_label"].value_counts().sort_index()
    for cls in class_counts.index:
        print_log(f"Samples in Class {cls}       : {class_counts[cls]}", log)
    print_log("=" * 35, log)



# ------------------ Confusion matrix ------------------
def generate_detector_confusion_and_agreement(df, class_cols, output_dir):
    agreement_final_decision = []
    agreement_correct_decision = []
    agreement_both = []

    for i, col in enumerate(class_cols, 1):
        preds = df[col].values
        true = df["correct_label"].values
        final = df["final_prediction"].values

        col = col.split("_")[-1]  # Extract the detector number from the column name

        # Confusion matrix
        cm = confusion_matrix(true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (0)", "Fake (1)"])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix - Detector {col}")
        plt.savefig(os.path.join(output_dir, "confusions", f"confusion_detector_{col}.png"))
        plt.close()

        # Agreement when detector and final both match ground truth
        match_with_final = ((preds == final)).sum()
        agreement_final_decision.append({
            "Detector": f"Detector {col}",
            "Agrees with Final decisions": match_with_final,
            "Agreement Ratio": match_with_final / len(df)
        })

        match_with_correct = ((preds == true)).sum()
        agreement_correct_decision.append({
            "Detector": f"Detector {col}",
            "Agrees with Correct": match_with_correct,
            "Agreement Ratio": match_with_correct / len(df)
        })
        
        match_with_both = ((preds == true) & (preds == final)).sum()
        agreement_both.append({
            "Detector": f"Detector {col}",
            "Agrees with Both": match_with_both,
            "Agreement Ratio": match_with_both / len(df)
        })


    agreement_final_df = pd.DataFrame(agreement_final_decision)
    agreement_final_df.to_csv(os.path.join(output_dir, "detector_agreement.csv"), index=False)
    agreement_final_df.to_excel(os.path.join(output_dir, "detector_agreement.xlsx"), index=False)

    agreement_correct_df = pd.DataFrame(agreement_correct_decision)
    agreement_correct_df.to_csv(os.path.join(output_dir, "detector_agreement_correct.csv"), index=False)
    agreement_correct_df.to_excel(os.path.join(output_dir, "detector_agreement_correct.xlsx"), index=False)

    agreement_both_df = pd.DataFrame(agreement_both)
    agreement_both_df.to_csv(os.path.join(output_dir, "detector_agreement_both.csv"), index=False)
    agreement_both_df.to_excel(os.path.join(output_dir, "detector_agreement_both.xlsx"), index=False)


    # Bar plot for agreements
    plt.figure(figsize=(12, 5))
    sns.barplot(data=agreement_final_df, x="Detector", y="Agreement Ratio")
    plt.title("Detector Agreement with Final Decision")
    plt.ylabel("Agreement Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detector_final_agreement_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=agreement_correct_df, x="Detector", y="Agreement Ratio")
    plt.title("Detector Agreement with Correct Decisions")
    plt.ylabel("Agreement Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detector_correct_agreement_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=agreement_both_df, x="Detector", y="Agreement Ratio")
    plt.title("Detector Agreement with Both")
    plt.ylabel("Agreement Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detector_both_agreement_plot.png"))
    plt.close()

    return agreement_final_df, agreement_correct_df, agreement_both_df



# ------------------ Evaluation Function ------------------
def evaluate_model(model, X_test, y_test, batch_size):
    device = next(model.parameters()).device  # Get the device the model is on
    loader = DataLoader(DetectorsOutputDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())   # move back to CPU before .numpy()
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    return y_true, y_pred



def get_clip_embeddings(args, base_features, clip_csv_path):
    clip_df = pd.read_csv(clip_csv_path)

    assert "image_path" in clip_df.columns, "CLIP CSV must contain 'image_path' column."

    # Find all embedding columns (anything not 'image_path')
    embedding_cols = [col for col in clip_df.columns if col != "image_path" and "keyword" not in col]

    # Keep only the top-k embedding columns
    top_k = int(args.top_k_clip)
    embedding_cols = embedding_cols[:top_k]

    # Function to safely parse a single embedding string
    def parse_embedding_string(s):
        try:
            return np.fromstring(s, sep=',', dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse embedding string: {s}") from e

    # Now process each embedding column
    parsed_embeddings = {}

    for col in embedding_cols:
        print_log(f"[INFO] Parsing embedding column: {col}", log)

        # Parse each string into a np.array
        parsed_column = clip_df[col].apply(parse_embedding_string)

            # Verify consistency of embedding dimensions
        embedding_lengths = parsed_column.apply(len)
        if embedding_lengths.nunique() != 1:
            raise ValueError(f"Inconsistent dimensions found in column {col}: {embedding_lengths.unique()}")

        parsed_embeddings[col] = parsed_column

        # Concatenate all expanded embeddings together
    final_embeddings_df = pd.concat(parsed_embeddings.values(), axis=1)


        # Final dataframe: image_path + all embeddings
        # processed_sample_df = pd.concat([clip_df[['image_path']], final_embeddings_df], axis=1)

    print_log(f"[OK] CLIP embeddings processed: {final_embeddings_df.shape}", log)
        # merge the embeddings with base_features for training


    embedding_columns = [col for col in final_embeddings_df.columns if col.startswith('embedding_')]
    final_embeddings_df = final_embeddings_df[embedding_columns].apply(lambda row: np.concatenate(row.values), axis=1).values

    base_features = np.array(base_features)

    print_log(final_embeddings_df.shape, log)
    print_log(base_features.shape, log)

    features = [np.hstack([emb, baseF]) for emb, baseF in zip(final_embeddings_df, base_features)]

        
    # Convert the merged list into a DataFrame
    features = pd.DataFrame(features)

    print_log(f"[OK] Features shape after merging CLIP: {features.shape}", log)
    return features

def test_model(args):
    # assert args.test_csv is not None, "Please provide --test_csv for testing."
    # assert args.model_checkpoint is not None, "Please provide --model_checkpoint path for loading the model."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_suffix = f"_CLIP_Top{args.top_k_clip}" if args.use_clip else ""
    args.test_output_dir = os.path.join(args.test_output_dir, f"experiment{clip_suffix}_{timestamp}")
    os.makedirs(args.test_output_dir, exist_ok=True)

    df_test = merge_dataset()

    # Extraction des features
    score_cols_test = [col for col in df_test.columns if 'score' in col]
    base_features_test = df_test[score_cols_test].values

    if args.use_clip:
        assert args.test_clip_csv_path is not None, "Please provide --test_clip_csv_path for CLIP evaluation."
        test_features = get_clip_embeddings(args, base_features_test, args.test_clip_csv_path)
    else:
        test_features = base_features_test

    test_features = np.array(test_features)

    if args.data_norm:
        scaler = StandardScaler()
        test_features = scaler.fit_transform(test_features)

    # Chargement du modèle
    model_ckpt = torch.load("/home/bellelbn/DL/Imverif/outputs_without_frequent_old_labels/experiment_CLIP_Top4_20250501_024317/meta_classifier.pth")
    model = MetaClassifier(input_dim=test_features.shape[1], hidden_dim=args.hidden_dim)
    model.load_state_dict(model_ckpt)

    # Prédictions
    y_pred, y_scores = predict_model(model, test_features, args.batch_size)

    # Sauvegarde des résultats
    df_results = pd.DataFrame({
        "image_path": df_test["image_path"],
        "score": y_scores,
        "predicted_label": y_pred
    })

    output_path = "output/final_predictions.csv"
    df_results.to_csv(output_path, index=False)

    print(f"✅ Prédictions enregistrées dans : {output_path}")


def predict_model(model, X_test, batch_size):
    device = next(model.parameters()).device
    dataset = DetectorsOutputDataset(X_test, np.zeros(len(X_test)))  # Dummy labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            y_pred.extend(probs.argmax(1).cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())  # probabilité classe "1" (fake)

    return y_pred, y_scores


def merge_dataset():
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

        print(f"➡️ Colonnes détectées dans {file}: {df.columns.tolist()}")
        print(df.head())

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
    # merged_df.to_csv("output/merged_predictions.csv", index=False)
    merged_df.drop_duplicates(inplace=True)
    print("🧪 Nombres d'images uniques :", merged_df["image_path"].nunique())
    print("📄 Nombre total de lignes :", len(merged_df))
    print("🧹 image_path problématiques :")
    print(merged_df["image_path"][merged_df["image_path"].str.contains("e-", na=False)])
    return merged_df

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", default="/medias/db/ImagingSecurity_misc/Collaborations/model_assembly/merged_test.csv", type=str, help="Path to test CSV file")
    parser.add_argument("--test_clip_csv_path", default="/medias/db/ImagingSecurity_misc/Collaborations/model_assembly/merged_clip_test.csv", type=str, help="Path to test CLIP CSV file")
    # parser.add_argument("--model_checkpoint", type=str, help="Path to the model checkpoint for testing")
    parser.add_argument("--use_clip", action="store_true", help="Use CLIP embeddings")
    parser.add_argument("--top_k_clip", default=3, type=str, help="Use the top k CLIP embeddings")
    parser.add_argument("--data_norm", action="store_true", help="Apply data normalization")
    parser.add_argument("--test_output_dir", type=str, default="./test_outputs", help="Directory to save test outputs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for the model")

    args = parser.parse_args()

    assert args.test_csv is not None, "Please provide --test_csv for testing."
    # assert args.model_checkpoint is not None, "Please provide --model_checkpoint path for loading the model."

    # if args.use_clip:
    #     assert args.test_clip_csv_path is not None, "Please provide --test_clip_csv_path for testing."
    
    test_model(args)
