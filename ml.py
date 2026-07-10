import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import os

# -------------------------------
# Step 1: Load CSV safely
# -------------------------------
csv_path = os.path.join(os.path.dirname(__file__), 'internship.csv')

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: internship.csv not found at {csv_path}")
    exit()

# Normalize column names
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
print("Columns in CSV after normalization:", df.columns.tolist())

# -------------------------------
# Step 2: Preprocess Internship Data
# -------------------------------
# Convert REQUIRED_SKILLS to array
df["SKILLS_ARRAY"] = df.get("REQUIRED_SKILLS", pd.Series([""]*len(df))).fillna("").apply(lambda x: [s.strip() for s in x.split(",")])

# Convert DURATION like "3 Months" -> 3
df["DURATION_NUM"] = df.get("DURATION", pd.Series([0]*len(df))).astype(str).str.extract(r"(\d+)").fillna(0).astype(float)

# Encode ELIGIBILITY as numeric
eligibility_encoder = LabelEncoder()
df["ELIGIBILITY_ENCODED"] = eligibility_encoder.fit_transform(df.get("ELIGIBILITY", pd.Series([""]*len(df))).astype(str))

# -------------------------------
# Step 3: Encode Skills as Binary Vectors
# -------------------------------
all_skills = sorted({skill for skills in df["SKILLS_ARRAY"] for skill in skills if skill})

def encode_skills(skill_list):
    stripped_skills = {s.strip() for s in skill_list}
    return [1 if skill in stripped_skills else 0 for skill in all_skills]

df["SKILLS_VECTOR"] = df["SKILLS_ARRAY"].apply(encode_skills)

# -------------------------------
# Step 4: Normalize Numeric Features
# -------------------------------
scaler = MinMaxScaler()
df[["DURATION_SCALED", "ELIGIBILITY_SCALED"]] = scaler.fit_transform(
    df[["DURATION_NUM", "ELIGIBILITY_ENCODED"]]
)

# -------------------------------
# Step 5: Build Weighted Feature Matrix
# -------------------------------
# Weighting: skills=5x, duration=3x, eligibility=2x
feature_matrix = np.array([
    list(np.array(sv)*5) + [dur*3, elig*2] 
    for sv, dur, elig in zip(df["SKILLS_VECTOR"], df["DURATION_SCALED"], df["ELIGIBILITY_SCALED"])
])

# -------------------------------
# Step 6: Train NearestNeighbors Model
# -------------------------------
knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_model.fit(feature_matrix)

# -------------------------------
# Step 7: Recommendation Function
# -------------------------------
def recommend_internships(student_skills, student_duration, student_eligibility, top_n=5):
    try:
        student_elig_encoded = eligibility_encoder.transform([student_eligibility])[0]
    except ValueError:
        student_elig_encoded = -1  # unseen eligibility

    # Normalize student features
    student_duration_scaled, student_elig_scaled = scaler.transform(
        [[student_duration, student_elig_encoded]]
    )[0]

    # Encode skills and apply weights
    student_vector = list(np.array(encode_skills(student_skills))*5) + \
                     [student_duration_scaled*3, student_elig_scaled*2]
    student_vector = np.array(student_vector).reshape(1, -1)

    # Find nearest internships
    distances, indices = knn_model.kneighbors(student_vector, n_neighbors=top_n)
    recommended = df.iloc[indices[0]].copy()
    max_dist = distances.max() if distances.max() > 0 else 1
    recommended["MATCH_SCORE"] = 100 - (distances[0]/max_dist*100)

    return recommended[[
        "TITLE", "COMPANY_NAME", "REQUIRED_SKILLS", "DURATION_NUM",
        "ELIGIBILITY", "MATCH_SCORE"
    ]]

# -------------------------------
# Step 8: Test Example
# -------------------------------
if __name__ == "__main__":
    student_skills = ["Python", "SQL"]
    student_duration = 4
    student_eligibility = "Bachelor + Intermediate"  # put text matching ELIGIBILITY column

    top_internships = recommend_internships(
        student_skills, student_duration, student_eligibility, top_n=5
    )
    print(top_internships)
