import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI NGO Matcher", layout="wide")

df = pd.read_csv("dataset/ngo_clean.csv")

country_coords = {
    "india": (20.5937, 78.9629),
    "nigeria": (9.0820, 8.6753),
    "uganda": (1.3733, 32.2903),
    "kenya": (-0.0236, 37.9062),
    "germany": (51.1657, 10.4515),
    "usa": (37.0902, -95.7129),
    "canada": (56.1304, -106.3468),
    "ghana": (7.9465, -1.0232),
    "south africa": (-30.5595, 22.9375)
}

df["primary_country"] = df["country_of_activity"].str.split("|").str[0].str.strip().str.lower()

def get_coords(country):
    return country_coords.get(country, (0, 0))

df["lat"] = df["primary_country"].apply(lambda x: get_coords(x)[0])
df["lon"] = df["primary_country"].apply(lambda x: get_coords(x)[1])

df["main_text"] = (
    (df["main_objective"].fillna('') + " ") * 3 +
    (df["area_of_expertise_note"].fillna('') + " ") * 2 +
    (df["country_of_activity"].fillna('') + " ")
).str.lower()

df["region_text"] = df["region"].fillna('').str.lower()

vectorizer_main = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3)
vectorizer_region = TfidfVectorizer()

tfidf_main = vectorizer_main.fit_transform(df["main_text"])
tfidf_region = vectorizer_region.fit_transform(df["region_text"])

ngo_similarity_matrix = cosine_similarity(tfidf_main)

def build_query(volunteer):
    skills_text = " ".join(volunteer["skills"]) * 2
    interest_text = volunteer["interest"] * 3
    location_text = volunteer["location"]
    return f"{skills_text} {interest_text} {location_text}".lower()

def map_location_to_region(location):
    location = location.lower()
    if location in ["india", "china", "japan"]:
        return "asia"
    elif location in ["nigeria", "kenya", "ghana"]:
        return "africa"
    elif location in ["usa", "canada"]:
        return "north america"
    else:
        return "asia"

def compute_skill_score(volunteer_skills, ngo_text):
    matches = sum(1 for skill in volunteer_skills if skill.lower() in ngo_text)
    return matches / len(volunteer_skills) if volunteer_skills else 0

def diversify_results(df_sorted, similarity_matrix, top_n=5, threshold=0.7):
    selected_indices = []
    for idx in df_sorted.index:
        if not selected_indices:
            selected_indices.append(idx)
            continue
        
        is_diverse = True
        for selected_idx in selected_indices:
            if similarity_matrix[idx][selected_idx] > threshold:
                is_diverse = False
                break
        
        if is_diverse:
            selected_indices.append(idx)
        
        if len(selected_indices) >= top_n:
            break
    
    return df_sorted.loc[selected_indices]

def match_volunteer_final(volunteer, top_n=5):
    query = build_query(volunteer)
    region = map_location_to_region(volunteer["location"])
    
    query_main = vectorizer_main.transform([query])
    query_region = vectorizer_region.transform([region])
    
    sim_main = cosine_similarity(query_main, tfidf_main).flatten()
    sim_region = cosine_similarity(query_region, tfidf_region).flatten()
    
    df_temp = df.copy()
    df_temp["semantic_score"] = sim_main
    df_temp["region_score"] = sim_region
    
    df_temp = df_temp[df_temp["region_score"] == 1.0]
    
    df_temp["skill_score"] = df_temp["main_text"].apply(
        lambda x: compute_skill_score(volunteer["skills"], x)
    )
    
    df_temp["final_score"] = (
        0.5 * df_temp["semantic_score"] +
        0.3 * df_temp["skill_score"] +
        0.2 * df_temp["region_score"]
    )
    
    df_temp = df_temp.sort_values(by="final_score", ascending=False)
    
    diversified = diversify_results(df_temp, ngo_similarity_matrix, top_n)
    
    return diversified[[
        "org_name",
        "region",
        "country_of_activity",
        "final_score",
        "lat",
        "lon"
    ]]

st.markdown("""
    <h1 style='text-align: center;'> AI Volunteer–NGO Matching System</h1>
    <p style='text-align: center; font-size:18px;'>Find NGOs tailored to your skills and interests</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    skills = st.text_input(" Skills (comma separated)")
    interest = st.text_input(" Interest")

with col2:
    location = st.text_input(" Location")
    st.write("")
    find = st.button("🔍 Find NGOs")

if find:
    volunteer = {
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "interest": interest,
        "location": location
    }
    
    results = match_volunteer_final(volunteer)
    
    st.markdown(" Top Matches")
    
    for _, row in results.iterrows():
        st.markdown(f"""
        {row['org_name']}
        {row['region']}  
         {row['country_of_activity']}  
         Score: {round(row['final_score'], 3)}
        """)
        st.markdown("---")
    
    st.markdown("NGO Locations")
    
    map_data = results.rename(columns={"lat": "latitude", "lon": "longitude"})
    st.map(map_data)
