import streamlit as st
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec

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

# Function to compute average embedding for a sequence
def sequence_to_avg_embedding(sequence, embedding_dict, valid_chars):
    sequence = str(sequence).upper()
    sequence = ''.join([char if char in valid_chars else '0' for char in sequence])
    if len(sequence) < 100:
        sequence = sequence.ljust(100, '0')
    else:
        sequence = sequence[:100]
    embeddings = np.array([embedding_dict.get(char, np.zeros(100)) for char in sequence])
    return np.mean(embeddings, axis=0)

# Function to convert a SMILES string to Mol2Vec embedding
def smiles_to_mol2vec_torch(smiles, model, device):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(model.vector_size, device=device)
        sentence = mol2alt_sentence(mol, radius=1)
        embeddings = [
            torch.tensor(model.wv[word], dtype=torch.float32, device=device) if word in model.wv.key_to_index
            else torch.zeros(model.vector_size, dtype=torch.float32, device=device)
            for word in sentence
        ]
        if embeddings:
            return torch.mean(torch.stack(embeddings), dim=0)
        else:
            return torch.zeros(model.vector_size, device=device)
    except Exception as e:
        print(f"Error with SMILES '{smiles}': {e}")
        return torch.zeros(model.vector_size, device=device)

# Function to combine embeddings and predict binding affinity
def predict_binding_affinity(smiles, protein_sequence):
    protein_embedding = sequence_to_avg_embedding(protein_sequence, alphabet_embedding_dict, valid_characters)
    smiles_embedding = smiles_to_mol2vec_torch(smiles, mol2vec_model, "cpu")
    combined_embedding = np.concatenate([smiles_embedding, protein_embedding])
    y_pred_log = keras_model.predict(combined_embedding.reshape(1, -1))
    return np.exp(y_pred_log) - 1

# Streamlit interface
st.title("AffPro - Binding Affinity Predictor")

# Tab for user inputs
input_mode = st.radio("Select input mode:", ["Manual Input", "Upload CSV"])

if input_mode == "Manual Input":
    smiles_input = st.text_input("Enter SMILES string:")
    protein_sequence_input = st.text_area("Enter Protein Sequence:")
    if st.button("Predict"):
        if smiles_input and protein_sequence_input:
            result = predict_binding_affinity(smiles_input, protein_sequence_input)
            st.success(f" ## Predicted Binding Affinity: {result[0][0]:.4f}")
        else:
            st.error("Please provide both SMILES and Protein Sequence!")

elif input_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with columns 'SMILES', 'Protein Sequence', and 'Real K_i':", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        if input_df.iloc[0].isnull().any():
            st.error("The first row of the uploaded file contains empty values. Please ensure all rows have valid data.")
        else:
            input_df = input_df.head(21)

            if 'SMILES' in input_df.columns and 'Protein Sequence' in input_df.columns and 'Real K_i' in input_df.columns:
                input_df['Predicted K_i'] = input_df.apply(
                    lambda row: predict_binding_affinity(row['SMILES'], row['Protein Sequence'])[0][0], axis=1
                )
                input_df['Difference (Real - Predicted)'] = input_df['Real K_i'] - input_df['Predicted K_i']

                st.write(" ### - Predictions (First 20 Rows):")
                st.dataframe(input_df[['SMILES', 'Protein Sequence', 'Real K_i', 'Predicted K_i', 'Difference (Real - Predicted)']])

                st.write("###  - Comparison of Real and Predicted \(K_i\) (First 20 Rows):")
                plt.figure(figsize=(10, 6))
                plt.plot(input_df.index, input_df['Real K_i'], label="Real \(K_i\)", marker='o')
                plt.plot(input_df.index, input_df['Predicted K_i'], label="Predicted \(K_i\)", marker='x')
                plt.xlabel("Sample Index")
                plt.ylabel("Binding Affinity (\(K_i\))")
                plt.title("Real vs Predicted \(K_i\) (First 20 Rows)")
                plt.ylim(0, max(input_df[['Real K_i', 'Predicted K_i']].max()) * 1.2)
                plt.legend()
                plt.grid()
                st.pyplot(plt)

                top_3_min_predictions = input_df.nsmallest(3, 'Predicted K_i')
                st.write("### - Top 3 Minimum Predicted \(K_i\):")
                st.dataframe(top_3_min_predictions[['SMILES', 'Protein Sequence', 'Predicted K_i']])

                input_df[' - Top 3 Minimum Predicted'] = input_df.index.isin(top_3_min_predictions.index)

                # Additional \(K_i\) Threshold Categorization
                st.write("###  - Categorize Predictions Based on \(K_i\) Thresholds")
                min_threshold = st.number_input("Enter Minimum \(K_i\) Threshold for Binding:", min_value=0.0, value=1.0, step=0.1)
                max_threshold = st.number_input("Enter Maximum \(K_i\) Threshold for Not Binding:", min_value=min_threshold, value=500.0, step=0.1)

                def categorize_ki(predicted_ki):
                    if predicted_ki <= min_threshold:
                        return "Binding"
                    elif predicted_ki >= max_threshold:
                        return "Not Binding"
                    else:
                        return "Moderate Binding"

                input_df['Category'] = input_df['Predicted K_i'].apply(categorize_ki)

                st.write("####  Categorized Predictions (Based on Thresholds):")
                st.dataframe(input_df[['SMILES', 'Protein Sequence', 'Predicted K_i', 'Category']])

                categorized_csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="#### Download Categorized Predictions as CSV",
                    data=categorized_csv,
                    file_name="categorized_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Uploaded CSV must contain 'SMILES', 'Protein Sequence', and 'Real K_i' columns.")


            