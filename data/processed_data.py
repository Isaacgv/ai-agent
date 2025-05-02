import pandas as pd
import zipfile
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === Load Data ===
df = pd.read_csv("/home/isaac/Downloads/wisky/combined_output.csv")

# === Load Descriptions from ZIP ===
descriptions = {}
with zipfile.ZipFile("descriptions.zip", 'r') as z:
    for file in z.namelist():
        if file.endswith('.txt'):
            with z.open(file) as f:
                whisky_name = int(file.split(".")[0])
                descriptions[whisky_name] = f.read().decode('utf-8')

df['description_text'] = df['id'].map(descriptions).fillna('')


# === Define All Text Features ===
text_features = ['Flavors', 'Nose', 'Palate', 'Finish', 'Mash Bill Type', 'Barrel Finish',
                 'Distillery', 'Region', 'Subregion', 'Proof Style', 'Drink Style', 'Best Pairing', 'Spice Level','Sweetness Level' ]
numeric_features = ['size', 'proof', 'abv', 'popularity', 'avg_msrp', 'fair_price', 'shelf_price',
                    'total_score', 'wishlist_count', 'vote_count', 'bar_count', 'ranking',
                    'Spice Level Numeric', 'Sweetness Level Numeric']
df[text_features] = df[text_features].fillna('').replace('Unknown', '')

# === Map Levels to Numeric ===
level_mapping = {
    'Low': 1,
    'Mild': 1.5,
    'Medium': 2,
    'Moderate': 2.5,
    'Medium-high': 3,
    'High': 3.5
}
df['Spice Level Numeric'] = df['Spice Level'].map(level_mapping).fillna(0)
df['Sweetness Level Numeric'] = df['Sweetness Level'].map(level_mapping).fillna(0)

# === Numeric Features for Scaling ===
numeric_features = ['size', 'proof', 'abv', 'popularity', 'avg_msrp', 'fair_price', 'shelf_price',
                    'total_score', 'wishlist_count', 'vote_count', 'bar_count', 'ranking',
                    'Spice Level Numeric', 'Sweetness Level Numeric']
df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === TF-IDF per field including description_text ===
text_vector_fields = ['description_text'] + text_features
vectorizers = {}
tfidf_matrices = []

for field in text_vector_fields:
    text_data = df[field].astype(str).fillna('').str.strip()
    
    if text_data.str.len().sum() == 0:
        print(f"⚠️ Skipping empty field: {field}")
        continue

     
    custom_stop_words = [
                "the", "and", "with", "this", "that", "from", "notes", "note", "finish", "palate", "also", "content",
                "flavor", "flavours", "taste", "aroma", "on", "a", "an", "is", "are", "it", "has", "text", "visual", "complete", 
                "overall", "based", "bold", "main", "features", "design", "catching", "made", 'bottle'
            ]

    # Combine with default English stop words
    all_stop_words = ENGLISH_STOP_WORDS.union(custom_stop_words)

    vectorizer = TfidfVectorizer(max_features=1000, stop_words=all_stop_words)
    try:
        tfidf_matrix = vectorizer.fit_transform(text_data)
        if tfidf_matrix.shape[1] == 0:
            print(f"⚠️ No valid tokens in field (stop words only?): {field}")
            continue
        vectorizers[field] = vectorizer
        tfidf_matrices.append(tfidf_matrix)
    except ValueError as e:
        print(f"❌ Skipping field {field} due to error: {e}")

# Combine all TF-IDF matrices into one
final_tfidf_matrix = hstack(tfidf_matrices)

# === Standard Scaling for Numeric Features ===
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(df[numeric_features])

# === Save all data and objects to pickle ===
with open("processed_data.pkl", "wb") as f:
    pickle.dump({
        "df": df,
        "vectorizers": vectorizers,
        "tfidf_matrix": final_tfidf_matrix,
        "scaler": scaler,
        "numeric_scaled": numeric_scaled,
        "numeric_features": numeric_features
    }, f)

print("✅ Preprocessing complete. File saved as: processed_data.pkl")
