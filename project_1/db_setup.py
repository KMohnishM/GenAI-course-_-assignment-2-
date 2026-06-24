# db_setup.py
import sqlite3
import pandas as pd
import os

def setup_db():
    csv_path = os.path.join('gold', 'clean_data.csv')
    db_path = 'student_performance.db'
    schema_path = 'schema.sql'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Please run the ETL pipeline first!")
        return
        
    print(f"Loading cleaned data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys support in SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Read and execute schema SQL
    print(f"Reading schema from {schema_path} and creating tables...")
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    
    # Clean up existing tables to allow re-runs
    print("Clearing old records from tables (if any)...")
    cursor.execute("DELETE FROM grades_absences;")
    cursor.execute("DELETE FROM student_lifestyle;")
    cursor.execute("DELETE FROM family_background;")
    cursor.execute("DELETE FROM students;")
    conn.commit()
    
    print("Inserting data into database tables...")
    for idx, row in df.iterrows():
        # 1. Insert into students table
        cursor.execute("""
            INSERT INTO students (school, sex, age, address, famsize, pstatus)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (row['school'], row['sex'], int(row['age']), row['address'], row['famsize'], row['pstatus']))
        
        student_id = cursor.lastrowid
        
        # 2. Insert into family_background table
        cursor.execute("""
            INSERT INTO family_background (student_id, medu, fedu, mjob, fjob, reason, guardian)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (student_id, int(row['medu']), int(row['fedu']), row['mjob'], row['fjob'], row['reason'], row['guardian']))
        
        # 3. Insert into student_lifestyle table
        cursor.execute("""
            INSERT INTO student_lifestyle (
                student_id, traveltime, studytime, failures, schoolsup, famsup, 
                paid, activities, nursery, higher, internet, romantic, 
                famrel, freetime, goout, health
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            student_id, int(row['traveltime']), int(row['studytime']), int(row['failures']),
            row['schoolsup'], row['famsup'], row['paid'], row['activities'], row['nursery'],
            row['higher'], row['internet'], row['romantic'], int(row['famrel']),
            int(row['freetime']), int(row['goout']), int(row['health'])
        ))
        
        # 4. Insert into grades_absences table
        cursor.execute("""
            INSERT INTO grades_absences (student_id, dalc, walc, absences, g1, g2, g3)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            student_id, int(row['dalc']), int(row['walc']), int(row['absences']),
            int(row['g1']), int(row['g2']), int(row['g3'])
        ))
        
    conn.commit()
    print("Data insertion completed successfully!")
    
    # Verify and print record counts
    print("\n--- Verification Summary ---")
    for table in ['students', 'family_background', 'student_lifestyle', 'grades_absences']:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' row count: {count}")
        
    conn.close()

if __name__ == '__main__':
    setup_db()
