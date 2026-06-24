# Mini Project 2: Software Development with Gen AI - Submission Draft

This document contains all prompts, screenshots/wireframe blueprints, SQL schema statements, code modules, and simulation results ready to compile into your report.

---

## PHASE 1: Requirements Analysis
**Tool Used:** ChatGPT

### Prompts Used:
1. *"Write 12 realistic feedback messages from imaginary patients and staff using an online hospital booking application. Cover issues like scheduling conflicts, billing clarity, search usability, and mobile responsiveness."*
2. *"Based on these feedback messages, extract a structured requirements table with ID, Description, Category (Functional/Non-Functional), and Priority (High/Medium/Low)."*
3. *"Write user stories in standard format for all High priority functional requirements."*
4. *"Identify 5 non-functional requirements critical for a hospital appointment system."*

### Simulated User Feedback (10-15 Messages):
1. *"I tried to book an appointment with a Cardiologist, but the calendar doesn't show which time slots are already taken. I got a database double-booking error."* - Patient
2. *"I need to see my invoice right after booking so I can submit it to my insurance. Right now I don't get any receipt."* - Patient
3. *"It would be great to see the doctor's experience level and consultation fee before booking. I am on a tight budget."* - Patient
4. *"The search button doesn't work well on my iPhone. The layout cuts off half the doctor list."* - Patient
5. *"As a clinic coordinator, I need to see a daily list of all booked appointments sorted by time so I can prepare the patient charts."* - Staff
6. *"I couldn't cancel my appointment online. I had to call the front desk and wait in a queue for 10 minutes."* - Patient
7. *"I am worried about my medical records. Is this database secure and HIPAA compliant?"* - Patient
8. *"I didn't receive any SMS or email confirmation after booking. I wasn't sure if my slot was actually reserved."* - Patient
9. *"It takes forever to load the doctor specialties page in the morning when everybody is logging in."* - Patient
10. *"I want to filter doctors not just by specialty, but also by their consultation fee."* - Patient
11. *"As a doctor, I want to set my own daily available hours so that patients don't book me when I'm in surgery."* - Doctor
12. *"The database crashes whenever we run the monthly billing report while patients are booking slots."* - Staff

### Structured Requirements Table:
| Requirement ID | Description | Category | Priority |
|---|---|---|---|
| REQ-01 | Prevent double-booking for the same doctor, date, and time slot. | Functional | High |
| REQ-02 | Automatically generate an unpaid billing invoice upon booking. | Functional | High |
| REQ-03 | Search and filter doctors by specialty and consultation fee. | Functional | High |
| REQ-04 | Display doctor details (experience years, fees, and specialty). | Functional | High |
| REQ-05 | List active appointments for a patient. | Functional | Medium |
| REQ-06 | Admin dashboard showing daily appointments. | Functional | Medium |
| REQ-07 | SSL encryption and database security controls for health records. | Non-Functional | High |
| REQ-08 | System response time must be under 300ms under normal load. | Non-Functional | High |

### User Stories:
1. **User Story 1 (REQ-01 - Avoid Double Booking):**
   * *As a* patient,
   * *I want to* only see and book time slots that are currently vacant,
   * *So that* I do not experience schedule overlaps or double-booking conflicts.
2. **User Story 2 (REQ-02 - Invoice Generation):**
   * *As a* patient,
   * *I want to* automatically receive a detailed invoice showing amount due,
   * *So that* I can track my expenses and submit them to insurance.
3. **User Story 3 (REQ-03 - Search/Filter):**
   * *As a* patient,
   * *I want to* filter doctors by their specialty and maximum consultation fee,
   * *So that* I can find a suitable specialist within my budget.

### Non-Functional Requirements:
1. **Security & Privacy (HIPAA):** All patient names, phone numbers, and emails must be encrypted at rest and in transit.
2. **Performance:** The API response time must remain below 300ms under normal operation.
3. **Availability:** The system must maintain 99.9% uptime.
4. **Data Integrity:** Database constraints must strictly enforce foreign key integrity and check ranges (e.g. positive fees, unique emails).
5. **Usability:** The booking interface must adapt responsively to mobile, tablet, and desktop viewports.

---

## PHASE 2: Wireframe Design
**Tool Used:** Visily

### AI Prompt Used:
*"Design a modern, medical-themed SaaS landing page and appointment booking dashboard named CarePulse. Use a clean slate-blue and teal layout. The design must feature: 
1) A responsive header with hospital logo and portal links, 
2) A search grid showing doctor profile cards with specialty, experience badge, pricing, and book button, 
3) A patient active appointments tracker table with color-coded status pills (Confirmed in teal, Pending in orange) and billing status badges (Paid in green, Unpaid in red)."*

### Wireframe Connections and Description:
* **Home Screen:** Features a search bar allowing specialty filters. Prominently displays the doctor profile card grid. Clicking the "Book" button on a doctor card opens the booking modal.
* **Booking Action:** A clean, centered modal form collects patient ID, appointment date, and time slot. Clicking "Confirm Booking" validates inputs, sends an API request, updates the database, and transitions to the summary screen.
* **Dashboard / Summary Screen:** Displays a patient status layout with a list of active appointments, doctor details, and a clear payment card showing the unpaid invoice amount due.

---

## PHASE 3: Architecture & Database Design
**Tool Used:** Eraser.io

### AI Prompt Used:
*"Create an Entity Relationship Diagram (ERD) and system architecture blueprint for CarePulse. The database is SQLite and must contain 4 normalized tables: patients (patient_id, name, email, phone, dob), doctors (doctor_id, name, specialty, experience_years, consultation_fee), appointments (appointment_id, patient_id, doctor_id, date, slot, status), and billing (billing_id, appointment_id, amount_due, status). Connect tables with primary and foreign keys. The system architecture should be a classic 3-tier layout: Client Browser, Python Flask API Backend, and SQLite DB."*

### Database CREATE TABLE Statements:
```sql
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT NOT NULL,
    dob TEXT NOT NULL
);

CREATE TABLE doctors (
    doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    experience_years INTEGER NOT NULL,
    consultation_fee REAL NOT NULL
);

CREATE TABLE appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    appointment_date TEXT NOT NULL,
    time_slot TEXT NOT NULL,
    status TEXT DEFAULT 'Pending',
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY(doctor_id) REFERENCES doctors(doctor_id)
);

CREATE TABLE billing (
    billing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    appointment_id INTEGER NOT NULL,
    amount_due REAL NOT NULL CHECK(amount_due >= 0),
    payment_status TEXT DEFAULT 'Unpaid',
    FOREIGN KEY(appointment_id) REFERENCES appointments(appointment_id)
);
```

---

## PHASE 4: Development
**Tool Used:** Google Colab (IPython.display.HTML)

*Please review `notebook.ipynb` for the full code. Below is the output validation.*

### Database Row Counts (Verification):
* Table `patients`: 3 records
* Table `doctors`: 5 records
* Table `appointments`: 3 records
* Table `billing`: 3 records

---

## PHASE 5: Testing Suite
**Tool Used:** Google Colab

The automated test suite runs 4 assertion test cases:
1. **Database Table Verification:** Checks that `patients`, `doctors`, `appointments`, and `billing` tables exist in `sqlite_master`.
2. **Data Quality Range Verification:** Ensures that checking negative billing amounts violates the SQLite range check constraint.
3. **Business Logic Verification:** Verifies that calling `book_appointment()` successfully creates an appointment in the database and generates an unpaid bill matching the doctor's fee.
4. **Edge Case Double-Booking Prevention:** Asserts that trying to book a doctor for an already reserved date and slot raises a `ValueError`.

**Test Execution Output:**
* `Test Database Structures... PASS`
* `Test Negative Invoice Quality... PASS`
* `Test Booking Logic & Billing creation... PASS`
* `Test Double Booking Prevention... PASS`
* **TOTAL PASS RATE: 100% (4/4 tests passed)**

---

## PHASE 6: Deployment & Monitoring
**Tool Used:** Google Colab (10-Minute Response Time Simulation)

### Output Log:
```
Minute 01: Avg Response Time = 143.52 ms
Minute 02: Avg Response Time = 162.11 ms
Minute 03: Avg Response Time = 155.80 ms
Minute 04: Avg Response Time = 1245.92 ms [ALERT] Response threshold breached (>500ms)!
           [AUTO-SCALE] Triggered horizontal scaling. Spinning up 2 API containers...
Minute 05: Avg Response Time = 1312.43 ms [ALERT] Response threshold breached (>500ms)!
           [CACHING] Routing doctor search queries to Redis cache...
Minute 06: Avg Response Time = 1198.54 ms [ALERT] Response threshold breached (>500ms)!
Minute 07: Avg Response Time = 184.20 ms  [RECOVERED] Response time back to normal.
Minute 08: Avg Response Time = 148.15 ms
Minute 09: Avg Response Time = 153.90 ms
Minute 10: Avg Response Time = 139.77 ms
```

---

## PHASE 7: Maintenance & Prediction
**Tool Used:** Google Colab (30-Day Server Metrics Prediction)

### Server Capacity Forecasting:
* **Metric Tracked:** Database CPU Load (%)
* **Prediction:** Based on the 30-day linear regression model, the Database Load is projected to breach the critical **90% threshold on day 31**.

### Prioritized Maintenance Recommendations:
1. **[CRITICAL] SQL Database Migration (Day 1 - 5):** Migrate CarePulse from SQLite to a fully-managed PostgreSQL database. SQLite locks the entire database during write operations, causing the spikes observed under simultaneous bookings.
2. **[HIGH] Add Database Indexing (Day 6 - 10):** Add indexing on `appointments(doctor_id, appointment_date, time_slot)` and `patients(email)` to speed up checking double-bookings and patient logins.
3. **[HIGH] Implement Redis Caching (Day 11 - 15):** Cache the doctor listings and schedule lists in memory to offload read-only queries from the primary SQL database.
4. **[MEDIUM] Setup Auto-Scaling Rules (Day 16 - 20):** Configure auto-scaling in Kubernetes or AWS based on CPU load and response time thresholds.
5. **[LOW] Archive Past Records (Quarterly):** Run a cron job to archive completed appointments and settled bills older than 1 year to a data lake, keeping the operational tables compact.

### Written Reflection:
"The prediction that concerns me the most is the database load hitting 90% in just 31 days. SQLite is a lightweight, file-based database, which is excellent for mockups and prototypes, but it lacks concurrency controls. In a real-world clinic, multiple patients book slots simultaneously. As the load grows, SQLite’s table-locking mechanism will result in write locks, API queues, and eventual server timeouts (HTTP 504), as simulated in Phase 6. Resolving this database constraint is our highest technical priority."
