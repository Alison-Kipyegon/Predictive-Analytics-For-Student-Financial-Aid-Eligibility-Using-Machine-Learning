import sqlite3
import pandas as pd
import os

# Path to dataset
CSV_PATH = "dataset/financial_aid_dataset.csv"
DB_PATH = "database/financial_aid.db"

# Delete old DB if it exists or corrupted
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Load dataset
df = pd.read_csv(CSV_PATH)

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop students table if it exists (to reset schema)
cursor.execute("DROP TABLE IF EXISTS students;")

# Create students table with correct schema
cursor.execute("""
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender TEXT,
    age INTEGER,
    study_hours INTEGER,
    attendance REAL,
    performance_score REAL,
    household_income_bracket TEXT,
    dependents INTEGER,
    background TEXT,
    government_assistance INTEGER,
    scholarship_eligibility INTEGER
);
""")

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    otp_secret TEXT NOT NULL,
    otp_verified INTEGER DEFAULT 0
);
""")

# Insert student data
df.to_sql("students", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("Database setup complete with students + users tables.")
