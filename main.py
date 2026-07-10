from flask import Flask, request, jsonify
import pymysql
import json
import os
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# Flask App Setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# MySQL connection
# -------------------------------
try:
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="gunjan@6490",
        database="intern",
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print("Connected to MySQL successfully.")
except pymysql.MySQLError as e:
    print(f"Error connecting to database: {e}")
    db = None

# -------------------------------
# JSON backup
# -------------------------------
JSON_FILE = "students.json"
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, "w") as f:
        json.dump([], f)

# -------------------------------
# Load Internship Dataset & ML Model
# -------------------------------
try:
    df = pd.read_csv("internship.csv")
    df.columns = [c.replace(' ', '_').replace('.', '') for c in df.columns]

    # Preprocess skills
    df["Skills_array"] = df["REQUIRED_SKILLS"].fillna("").apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])

    # Encode eligibility (used as a proxy for experience/education)
    eligibility_map = {"Undergraduates": 0, "Graduates": 1, "Postgraduates": 2}
    df["ELIGIBILITY_ENCODED"] = df["ELIGIBILITY"].map(eligibility_map).fillna(0)
    df["Experience_encoded"] = df["ELIGIBILITY_ENCODED"]
    df["Education_encoded"] = df["ELIGIBILITY_ENCODED"]

    # Extract numeric duration
    def extract_duration(dur):
        if "Above" in dur:
            return 7.0
        parts = [float(s) for s in dur.split() if s.replace('.', '', 1).isdigit()]
        return sum(parts) / len(parts) if parts else 0.0

    df["Duration_num"] = df["DURATION"].apply(extract_duration)

    # Skills vector
    all_skills = sorted({skill for skills in df["Skills_array"] for skill in skills})
    def encode_skills(skill_list):
        return [1 if skill in skill_list else 0 for skill in all_skills]
    df["Skills_vector"] = df["Skills_array"].apply(encode_skills)

    # Normalize numeric features
    scaler = MinMaxScaler()
    df[["Duration_scaled", "Experience_scaled", "Education_scaled"]] = scaler.fit_transform(
        df[["Duration_num", "Experience_encoded", "Education_encoded"]]
    )

    # Weighted feature matrix: skills=5x, duration=3x, experience=1x, education=1x
    feature_matrix = np.array([
        list(np.array(sv)*5) + [dur*3, exp*1, edu*1]
        for sv, dur, exp, edu in zip(df["Skills_vector"], df["Duration_scaled"], df["Experience_scaled"], df["Education_scaled"])
    ])

    # Train KNN model
    knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn_model.fit(feature_matrix)

    ml_ready = True
    print("ML model loaded and trained successfully.")
except Exception as e:
    print(f"Error loading ML model or data: {e}")
    ml_ready = False

# -------------------------------
# Add Student API
# -------------------------------
@app.route("/add_student", methods=["POST"])
def add_student():
    if db is None:
        return jsonify({"message": "Database connection failed"}), 500

    data = request.json
    required_fields = ["name", "education_level", "experience_level", "skills", "preferred_duration"]
    if not all(field in data and data.get(field) is not None for field in required_fields):
        return jsonify({"message": "Missing one or more required fields"}), 400

    try:
        skills_json = json.dumps(data.get("skills", []))
        cursor = db.cursor()
        query = """
        INSERT INTO student (NAME, EDUCATION_LEVEL, EXPERIENCE_LEVEL, SKILLS, PREFERRED_DURATION)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (data["name"], data["education_level"], data["experience_level"], skills_json, data["preferred_duration"]))
        db.commit()
        cursor.close()

        # Save in JSON
        with open(JSON_FILE, "r+") as f:
            try:
                students = json.load(f)
            except json.JSONDecodeError:
                students = []
            students.append(data)
            f.seek(0)
            json.dump(students, f, indent=4)
            f.truncate()

        return jsonify({"message": "✅ Student data saved in MySQL + JSON!"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# -------------------------------
# ML Recommendation API
# -------------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    if not ml_ready:
        return jsonify({"message": "Recommendation model is not ready."}), 503

    try:
        data = request.json
        user_skills = set(data.get("skills", []))
        user_experience = data.get("experience_level", "Beginner")
        user_education = data.get("education_level", "Bachelor")
        user_duration = data.get("preferred_duration", "1-3 Months")

        # Encode experience & education
        experience_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        education_map = {"High School": 0, "Associate": 0, "Bachelor": 1, "Master": 2, "PhD": 2}
        exp_encoded = experience_map.get(user_experience, 0)
        edu_encoded = education_map.get(user_education, 0)

        # Duration numeric
        def dur_to_num(dur):
            if "Above" in dur:
                return 7.0
            parts = [float(s) for s in dur.split() if s.replace('.', '', 1).isdigit()]
            return sum(parts) / len(parts) if parts else 0.0
        dur_encoded = dur_to_num(user_duration)

        # Encode skills vector
        student_skills_vector = list(np.array(encode_skills(user_skills)) * 5)
        student_vector_scaled = scaler.transform([[dur_encoded, exp_encoded, edu_encoded]])[0]
        student_vector = np.array(student_skills_vector + [student_vector_scaled[0]*3, student_vector_scaled[1]*1, student_vector_scaled[2]*1]).reshape(1, -1)

        distances, indices = knn_model.kneighbors(student_vector, n_neighbors=5)
        recommended = df.iloc[indices[0]].copy()

        max_dist = distances.max() if distances.max() > 0 else 1
        recommended["Match_Score"] = 100 - (distances[0][:len(recommended)] / max_dist * 100)

        output = recommended[["TITLE", "COMPANY_NAME", "REQUIRED_SKILLS", "DURATION", "ELIGIBILITY", "Match_Score"]].rename(columns={
            "REQUIRED_SKILLS": "skills",
            "DURATION": "duration",
            "ELIGIBILITY": "eligibility"
        }).to_dict(orient="records")

        return jsonify(output)
    except Exception as e:
        print(f"Error in recommend function: {e}")
        return jsonify({"message": "Error processing recommendation request"}), 500

# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
