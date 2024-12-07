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

# Step 3: Define a function to check and compute the average embedding for a sequence
import numpy as np

def sequence_to_avg_embedding(sequence, embedding_dict, valid_chars):
    """
    Converts a protein sequence into a single 100-dimensional vector
    by averaging the embeddings of all characters in the sequence.
    If any invalid character is found, it is replaced by '0'.

    :param sequence: Protein sequence string.
    :param embedding_dict: Dictionary mapping alphabets to 100-dimensional embeddings.
    :param valid_chars: Set of valid characters based on the embedding file.
    :return: Numpy array of average embeddings for the sequence.
    """
    # Convert sequence to uppercase for case-insensitive processing
    sequence = str(sequence).upper()

    # Step 1: Handle invalid characters by replacing them with '0'
    sequence = ''.join([char if char in valid_chars else '0' for char in sequence])

    # Step 2: Adjust sequence length
    if len(sequence) < 100:
        sequence = sequence.ljust(100, '0')  # Pad with '0' if shorter than 100
    else:
        sequence = sequence[:100]  # Truncate to the first 100 characters if longer than 100

    # Step 3: Compute embeddings for the sequence
    embeddings = np.array([embedding_dict.get(char, np.zeros(100)) for char in sequence])

    # Return the average of embeddings
    return np.mean(embeddings, axis=0)






# Function to convert a single SMILES to Mol2Vec embedding
# Function to convert a single SMILES to Mol2Vec embedding
def smiles_to_mol2vec_torch(smiles, model, device):
    """
    Converts a single SMILES string into a Mol2Vec embedding using the pre-trained Word2Vec model.
    Args:
        smiles (str): SMILES string of a molecule.
        model (Word2Vec): Pre-trained Mol2Vec Word2Vec model.
        device (str): Device to perform the computation ('cpu' or 'cuda').
    Returns:
        torch.Tensor: 300-dimensional Mol2Vec embedding.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(model.vector_size, device=device)  # Return a zero tensor for invalid SMILES
        sentence = mol2alt_sentence(mol, radius=1)

        # Generate embedding by averaging vectors
        embeddings = [
            torch.tensor(model.wv[word], dtype=torch.float32, device=device) if word in model.wv.key_to_index
            else torch.zeros(model.vector_size, dtype=torch.float32, device=device)
            for word in sentence
        ]
        if embeddings:
            return torch.mean(torch.stack(embeddings), dim=0)
        else:
            return torch.zeros(model.vector_size, device=device)  # Return zero tensor if no embeddings generated
    except Exception as e:
        print(f"Error with SMILES '{smiles}': {e}")
        return torch.zeros(model.vector_size, device=device)





# Function to combine embeddings and predict binding affinity
def predict_binding_affinity(smiles, protein_sequence):

    protein_embedding = sequence_to_avg_embedding(protein_sequence, alphabet_embedding_dict, valid_characters)
    print("P Embeddings",protein_embedding)
    smiles_embedding = smiles_to_mol2vec_torch(smiles, mol2vec_model, "cpu")
    print("Smiles Embeddings: ",smiles_embedding )
    combined_embedding = np.concatenate([smiles_embedding, protein_embedding])
    print("Full Embeddings: ",combined_embedding)
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

                # Set y-axis limits to reduce fluctuations
                plt.ylim(0, max(input_df[['Real K_i', 'Predicted K_i']].max()) * 1.2)  # Adjust as needed

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