import streamlit as st
import pandas as pd
import pickle
import requests

from sklearn.metrics.pairwise import cosine_similarity
from langchain.agents import initialize_agent, AgentType, tool
from langchain.llms import OpenAI
from scipy.sparse import hstack

import re
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Whisky Recommender", layout="wide")

openai_api_key = os.getenv("OPENAI_API_KEY")

# Load backend preprocessed data
@st.cache_resource
def load_processed_data():
    with open("data/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data

data = load_processed_data()
df = data["df"]
tfidf_matrix = data["tfidf_matrix"]
vectorizers = data["vectorizers"]
scaler = data["scaler"]
numeric_scaled = data["numeric_scaled"]
numeric_features = data["numeric_features"]


st.title("Meet Bob — Your Personalized Whisky Concierge")
st.markdown('''
**Bob** is your personal AI whisky expert inside the BAXUS ecosystem.  
Trained on extensive whisky data and bottle profiles, Bob analyzes your virtual bar to deeply understand your unique palate.

### Key Features:

- 🧠 **AI-powered taste profiling** from your existing bar  
- 🥃 **Personalized bottle suggestions** for your wishlist  
- 📊 **Insights into your whisky preferences** and collection style  
- 🌍 **Explore new bottles** by flavor, region, rarity, or user similarity
''')


# Recommendation function 
def get_recommendations(df, whiskies, intent='similar', top_n=5, sort_by="Highest similarity"):
    user_df = df[df['name'].isin(whiskies)]

    user_text_parts = []
    all_text_parts = []

    for field, vectorizer in vectorizers.items():
        user_text = user_df[field].astype(str).fillna('')
        all_text = df[field].astype(str).fillna('')
        user_matrix = vectorizer.transform(user_text)
        all_matrix = vectorizer.transform(all_text)
        user_text_parts.append(user_matrix)
        all_text_parts.append(all_matrix)

    user_tfidf_matrix = hstack(user_text_parts)
    full_tfidf_matrix = hstack(all_text_parts)
    text_sim = cosine_similarity(user_tfidf_matrix, full_tfidf_matrix)

    user_numeric_scaled = scaler.transform(user_df[numeric_features])
    num_sim = cosine_similarity(user_numeric_scaled, numeric_scaled)
   
    similarity = (text_sim + num_sim) / 2
    df['similarity'] = similarity.mean(axis=0)

  
    if intent == "price":
        price = user_df['avg_msrp'].mean()
        filtered = df[df['avg_msrp'].between(price * 0.8, price * 1.2)]
    else:
        filtered = df

    # Exclude already selected whiskies
    recommendations = filtered[~filtered['name'].isin(whiskies)].copy()


    def extract_keywords(text, stopwords=None):
        if stopwords is None:
            stopwords = set([
                "the", "and", "with", "this", "that", "from", "notes", "note", "finish", "palate", "also", "content",
                "flavor", "flavours", "taste", "aroma", "on", "a", "an", "is", "are", "it", "has", "text", "visual", "complete", 
                "overall", "based", "bold", "main", "features", "design", "catching", "made"
            ])
        words = re.findall(r'\b\w+\b', text.lower())
        return set(w for w in words if w not in stopwords and len(w) > 3)

    def has_overlap(text1, text2):
        return bool(set(text1.lower().split()) & set(text2.lower().split()))

    # Generate reason for each recommendation
    def format_reason(row, user_df):
        reasons = []

        distillery_matches = user_df[user_df['Distillery'] == row.get('Distillery')]
        if not distillery_matches.empty:
            count = len(distillery_matches)
            bottles = ', '.join(distillery_matches['name'].tolist())
            reasons.append(f"Same distillery ({row.get('Distillery')}, {count} match{'es' if count > 1 else ''}: {bottles})")

        region_matches = user_df[user_df['Region'] == row.get('Region')]

        if not region_matches.empty and pd.notna(row.get('Region')):
            count = len(region_matches)
            reasons.append(f"Same region ({row.get('Region')}, {count} match{'es' if count > 1 else ''})")

        close_price_bottles = user_df[abs(user_df['avg_msrp'] - row.get('avg_msrp', 0)) < 10]['name'].tolist()

        if close_price_bottles:
            reasons.append(f"Similar price to: {', '.join(close_price_bottles)}")

        similar_proof_bottles = user_df[abs(user_df['proof'] - row.get('proof', 0)) < 5]['name'].tolist()

        if similar_proof_bottles:
            reasons.append(f"Similar proof to: {', '.join(similar_proof_bottles)}")

        if pd.notna(row.get('Barrel Finish')) and row.get('Barrel Finish') in user_df['Barrel Finish'].dropna().unique():
            reasons.append(f"Same barrel finish: {row.get('Barrel Finish')}")

        if pd.notna(row.get('Mash Bill Type')) and row.get('Mash Bill Type') in user_df['Mash Bill Type'].dropna().unique():
            reasons.append(f"Same mash bill type: {row.get('Mash Bill Type')}")

        shared_notes = []

        for field in ['Flavors', 'Nose', 'Palate', 'Finish']:
            user_text = " ".join(user_df[field].dropna().astype(str).tolist())

            if has_overlap(row.get(field, ''), user_text):
                shared_notes.append(field)
        if shared_notes:
            reasons.append(f"Shared tasting notes in: {', '.join(shared_notes)}")

        if pd.notna(row.get('Best Pairing')) and row.get('Best Pairing') in user_df['Best Pairing'].dropna().unique():
            reasons.append(f"Pairs well with similar foods: {row.get('Best Pairing')}")

        spice_diff = abs(row.get('Spice Level Numeric', 0) - user_df['Spice Level Numeric'].mean())
        sweetness_diff = abs(row.get('Sweetness Level Numeric', 0) - user_df['Sweetness Level Numeric'].mean())

        if spice_diff < 0.5:
            reasons.append("Similar spice level")

        if sweetness_diff < 0.5:
            reasons.append("Similar sweetness level")

        row_keywords = extract_keywords(str(row.get('description_text', '')))
        matches = []

        for _, user_row in user_df.iterrows():
            user_keywords = extract_keywords(str(user_row.get('description_text', '')))
            shared = row_keywords & user_keywords

            if shared:
                matches.append((user_row['name'], shared))
        if matches:
            desc_matches = [f"{name} (e.g. {', '.join(list(words)[:3])})" for name, words in matches[:2]]
            reasons.append("Similar description to: " + "; ".join(desc_matches))

        if row.get('similarity', 0) > 0.6:
            reasons.append("High flavor profile match")

        elif row.get('similarity', 0) > 0.4 and not reasons:
            reasons.append(f"Moderate profile similarity (score: {row.get('similarity'):.2f})")

        return ", ".join(reasons)

    filtered['reason'] = filtered.apply(lambda row: format_reason(row, user_df), axis=1)

    if sort_by == "Highest similarity":
        filtered = filtered.sort_values(by="similarity", ascending=False)

    elif sort_by == "Closest price":
        user_price = user_df['avg_msrp'].mean()
        filtered['price_diff'] = abs(filtered['avg_msrp'] - user_price)
        filtered = filtered.sort_values(by="price_diff")

    elif sort_by == "Same distillery":
        user_dists = user_df['Distillery'].unique()
        filtered['distillery_match'] = filtered['Distillery'].isin(user_dists)
        filtered = filtered.sort_values(by="distillery_match", ascending=False)

    return filtered[[
        'name', 'Distillery', 'Region', 'Flavors', 'Nose', 'Palate', 'Finish', 'Mash Bill Type', 'Barrel Finish',
        'Best Pairing', 'Spice Level Numeric', 'Sweetness Level Numeric', 'description_text', 'avg_msrp', 'proof',
        'similarity', 'reason', 'abv', 'Distillery', 'Spice Level','Sweetness Level']].head(top_n)

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


@tool("explanation_generator")
def explanation_generator(input: str) -> str:
    """
    Generate a natural-language explanation for why a whisky was recommended,
    based on user preferences and whisky characteristics. Return maximun two paragraph.
    """
    return llm.predict(input)


@tool("user_collection_retrieval")
def user_collection_retrieval(username: str):
    """Retrieve a user's whisky collection from the Baxus API."""
    response = requests.get(f"http://services.baxus.co/api/bar/user/{username}")
    if response.ok:
        return [item['product']['name'] for item in response.json() if item.get('product')]
    return []


@tool("whisky_recommendation_tool")
def whisky_recommendation_tool(query: str):
    """Provide whisky recommendations based on user's selected bottles and intent."""
    intent, whiskies = query.split(';')
    whiskies = [w.strip() for w in whiskies.split(',')]
    recs = get_recommendations(df, whiskies, intent.strip())
    return recs.to_markdown(index=False)

agent_tools = [
    whisky_recommendation_tool,
    user_collection_retrieval,
    explanation_generator 
]

agent = initialize_agent(agent_tools, llm, AgentType.OPENAI_FUNCTIONS, verbose=True)


st.subheader("🤖 BOB AI Agent Recommendations")
username = st.text_input("Enter your username (to use your bar data)")

sort_by = st.selectbox("Sort recommendations by", ["Highest similarity", "Closest price", "Same distillery"])
top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if username:
    try:
        with st.spinner("Fetching your bar data..."):
            whiskies = user_collection_retrieval(username)
            
        intent = 'similar'
        with st.spinner("Analyzing your collection and generating recommendations..."):
            recommendations = get_recommendations(df, whiskies, intent=intent, top_n=top_n, sort_by=sort_by)
        st.markdown("### 🥃 Recommended Bottles")
        for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
            name = str(row['name'])
            distillery = str(row['Distillery'])
            avg_msrp = float(row['avg_msrp'])
            similarity = float(row['similarity'])
            reason = str(row['reason'])
            card_html = f"""
            <div style="
                border: 1px solid #ccc; 
                border-radius: 10px; 
                padding: 16px; 
                margin-bottom: 16px; 
                background-color: #f9f9f9;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
                <h4>{i}. {name}</h4>
                <ul style="list-style: none; padding-left: 0;">
                    <li>🏭 <strong>Distillery:</strong> {distillery}</li>
                    <li>💲 <strong>Avg MSRP:</strong> ${avg_msrp:.2f}</li>
                    <li>📈 <strong>Similarity Score:</strong> <code>{similarity:.2f}</code></li>
                    <li>💡 <strong>Why:</strong> <em>{reason}</em></li>
                </ul>
            """
            prompt = f"""
            The user likes: {', '.join(whiskies)}.
            Why is "{row['name']}" a good recommendation?
            Traits:
            - Region: {row['Region']}
            - Distillery: {row['Distillery']}
            - Price: ${row['avg_msrp']:.2f}
            - ABV: {row.get('abv', 'Unknown')}%
            - Proof: {row.get('proof', 'Unknown')}
            - Flavor: {row.get('Flavors', '')}
            - Nose: {row.get('Nose', '')}
            - Palate: {row.get('Palate', '')}
            - Finish: {row.get('Finish', '')}
            - Mash Bill Type: {row.get('Mash Bill Type', '')}
            - Barrel Finish: {row.get('Barrel Finish', '')}
            - Best Pairing: {row.get('Best Pairing', '')}
            - Distillery: {row.get('Distillery', '')}
            - Spice Level: {row.get('Spice Level', '')}
            - Sweetness Level: {row.get('Sweetness Level', '')}
            """
            explanation = explanation_generator(prompt)
            card_html += f"<p><strong>🧠 Explanation:</strong> {explanation}</p></div>"
            st.markdown(card_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

