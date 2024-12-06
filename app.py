import streamlit as st
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import matplotlib.pyplot as plt
import time

# Paths to required models and files
ALPHABET_EMBEDDINGS_PATH = "models/embdedding_seq_2.csv" 
MOL2VEC_MODEL_PATH = "models/model_300dim.pkl"
KERAS_MODEL_PATH = "models/100_epochs_model.keras"

# Load alphabet embeddings
protein_data = pd.read_csv(ALPHABET_EMBEDDINGS_PATH)
alphabet_embedding_dict = protein_data.set_index('embedding').T.to_dict(orient='list')
valid_characters = set(alphabet_embedding_dict.keys())

# Load Mol2Vec model
mol2vec_model = Word2Vec.load(MOL2VEC_MODEL_PATH)

# Load the pre-trained Keras model
keras_model = load_model(KERAS_MODEL_PATH)

# Device configuration for PyTorch
device = "cpu"

# Function to process protein sequence
def process_protein_sequence(sequence):
    if pd.isnull(sequence):
        sequence = ""
    sequence = str(sequence).upper()
    return sequence[:100] if len(sequence) > 100 else sequence.ljust(100, "0")

# Function to convert a protein sequence into an average embedding
def sequence_to_avg_embedding(sequence, embedding_dict, valid_chars):
    sequence = str(sequence).upper()
    if any(char not in valid_chars for char in sequence):
        return np.zeros(100)
    embeddings = np.array([embedding_dict[char] for char in sequence])
    print(embeddings)
    return np.mean(embeddings, axis=0)

# Function to convert SMILES to Mol2Vec embeddings
def mol2alt_sentence(mol, radius):
    from rdkit.Chem.AllChem import GetMorganFingerprint
    info = {}
    GetMorganFingerprint(mol, radius, bitInfo=info)
    words = []
    for key, value in info.items():
        for atom_id, radius in value:
            words.append(f"{key}_{radius}_{atom_id}")
    return words

def smiles_to_mol2vec_torch(smiles, model, device):
    try:
        mol = MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(model.vector_size, device=device)
        sentence = mol2alt_sentence(mol, radius=1)
        embeddings = [
            torch.tensor(model.wv[word], dtype=torch.float32, device=device) if word in model.wv.key_to_index
            else torch.zeros(model.vector_size, dtype=torch.float32, device=device)
            for word in sentence
        ]
        print(embeddings)
        return torch.mean(torch.stack(embeddings), dim=0) if embeddings else torch.zeros(model.vector_size, device=device)
    except:
        return torch.zeros(model.vector_size, device=device)

# Function to combine embeddings and predict binding affinity
def predict_binding_affinity(smiles, protein_sequence):
    processed_protein = process_protein_sequence(protein_sequence)
    protein_embedding = sequence_to_avg_embedding(processed_protein, alphabet_embedding_dict, valid_characters)
    smiles_embedding = smiles_to_mol2vec_torch(smiles, mol2vec_model, device).cpu().numpy()
    combined_embedding = np.concatenate([protein_embedding, smiles_embedding])
    y_pred_log = keras_model.predict(combined_embedding.reshape(1, -1))
    return np.exp(y_pred_log) - 1

# Streamlit interface
st.title("AffPro- Binding Affinity Predictor ")

# Tab for user inputs
input_mode = st.radio("Select input mode:", ["Manual Input", "Upload CSV"])

if input_mode == "Manual Input":
    smiles_input = st.text_input("Enter SMILES string:")
    protein_sequence_input = st.text_area("Enter Protein Sequence:")
    if st.button("Predict"):
        if smiles_input and protein_sequence_input:
            result = predict_binding_affinity(smiles_input, protein_sequence_input)
            st.success(f"Predicted Binding Affinity: {result[0][0]:.4f}")
        else:
            st.error("Please provide both SMILES and Protein Sequence!")

elif input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with columns 'SMILES', 'Protein Sequence', and 'Real K_i':", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Check if the first row (or any row) has missing values
        if input_df.iloc[0].isnull().any():
            st.error("The first row of the uploaded file contains empty values. Please ensure all rows have valid data.")
        else:
            # Process only the first 20 rows
            input_df = input_df.head(21)

            # Ensure required columns are present
            if 'SMILES' in input_df.columns and 'Protein Sequence' in input_df.columns and 'Real K_i' in input_df.columns:
                input_df['Protein Sequence'] = input_df['Protein Sequence'].apply(process_protein_sequence)
                input_df['Predicted K_i'] = input_df.apply(
                    lambda row: predict_binding_affinity(row['SMILES'], row['Protein Sequence'])[0][0], axis=1
                )
                input_df['Difference (Real - Predicted)'] = input_df['Real K_i'] - input_df['Predicted K_i']

                # Display predictions
                st.write("Predictions (First 20 Rows):")
                st.dataframe(input_df[['SMILES', 'Protein Sequence', 'Real K_i', 'Predicted K_i', 'Difference (Real - Predicted)']])

                # Line plot for Real and Predicted \(K_i\)
                st.write("Comparison of Real and Predicted \(K_i\) (First 20 Rows):")
                plt.figure(figsize=(10, 6))
                plt.plot(input_df.index, input_df['Real K_i'], label="Real \(K_i\)", marker='o')
                plt.plot(input_df.index, input_df['Predicted K_i'], label="Predicted \(K_i\)", marker='x')
                plt.xlabel("Sample Index")
                plt.ylabel("Binding Affinity (\(K_i\))")
                plt.title("Real vs Predicted \(K_i\) (First 20 Rows)")
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                # Download results
                csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predicted_ki_values.csv",
                    mime="text/csv",
                )
            else:
                st.error("Uploaded CSV must contain 'SMILES', 'Protein Sequence', and 'Real K_i' columns.")



