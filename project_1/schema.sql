-- schema.sql
-- Normalized SQLite database schema for Student Performance dataset

-- 1. Student Demographic Table
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school TEXT NOT NULL,
    sex TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age BETWEEN 15 AND 22),
    address TEXT NOT NULL,
    famsize TEXT NOT NULL,
    pstatus TEXT NOT NULL
);

-- 2. Student Family Background Table
CREATE TABLE IF NOT EXISTS family_background (
    student_id INTEGER PRIMARY KEY,
    medu INTEGER NOT NULL CHECK (medu BETWEEN 0 AND 4),
    fedu INTEGER NOT NULL CHECK (fedu BETWEEN 0 AND 4),
    mjob TEXT NOT NULL,
    fjob TEXT NOT NULL,
    reason TEXT NOT NULL,
    guardian TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 3. Student Lifestyle and Habits Table
CREATE TABLE IF NOT EXISTS student_lifestyle (
    student_id INTEGER PRIMARY KEY,
    traveltime INTEGER NOT NULL CHECK (traveltime BETWEEN 1 AND 4),
    studytime INTEGER NOT NULL CHECK (studytime BETWEEN 1 AND 4),
    failures INTEGER NOT NULL CHECK (failures BETWEEN 0 AND 4),
    schoolsup TEXT NOT NULL,
    famsup TEXT NOT NULL,
    paid TEXT NOT NULL,
    activities TEXT NOT NULL,
    nursery TEXT NOT NULL,
    higher TEXT NOT NULL,
    internet TEXT NOT NULL,
    romantic TEXT NOT NULL,
    famrel INTEGER NOT NULL CHECK (famrel BETWEEN 1 AND 5),
    freetime INTEGER NOT NULL CHECK (freetime BETWEEN 1 AND 5),
    goout INTEGER NOT NULL CHECK (goout BETWEEN 1 AND 5),
    health INTEGER NOT NULL CHECK (health BETWEEN 1 AND 5),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- 4. Student Grades and Absences Table
CREATE TABLE IF NOT EXISTS grades_absences (
    student_id INTEGER PRIMARY KEY,
    dalc INTEGER NOT NULL CHECK (dalc BETWEEN 1 AND 5),
    walc INTEGER NOT NULL CHECK (walc BETWEEN 1 AND 5),
    absences INTEGER NOT NULL CHECK (absences >= 0),
    g1 INTEGER NOT NULL CHECK (g1 BETWEEN 0 AND 20),
    g2 INTEGER NOT NULL CHECK (g2 BETWEEN 0 AND 20),
    g3 INTEGER NOT NULL CHECK (g3 BETWEEN 0 AND 20),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
