import pandas as pd
from datetime import datetime

# ----------------------------
# 1. Load the CSV Files
# ----------------------------
patients = pd.read_csv('data/patients.csv/patients.csv')         # contains subject_id, gender, dob, etc.
admissions = pd.read_csv('data/admissions.csv/admissions.csv')       # contains subject_id, hadm_id, admittime, dischtime, etc.
# Explicitly assign column names for the diagnoses file (no header provided)
diagnoses = pd.read_csv('data/diagnosis.csv', encoding='latin1', on_bad_lines='skip', header=None,
                        names=['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'])
prescriptions = pd.read_csv('data/prescriptions.csv/prescriptions.csv') # contains hadm_id, drug, dosage, etc.
labevents = pd.read_csv('data/labevents.csv/labevents.csv')         # contains hadm_id, itemid, value, valuenum, etc.

# Ensure merge keys are the same type (convert to string for consistency)
admissions['hadm_id'] = admissions['hadm_id'].astype(str)
diagnoses['hadm_id'] = diagnoses['hadm_id'].astype(str)
prescriptions['hadm_id'] = prescriptions['hadm_id'].astype(str)
labevents['hadm_id'] = labevents['hadm_id'].astype(str)

# ----------------------------
# 2. Merge the DataFrames
# ----------------------------
# Merge patients with admissions on 'subject_id'
df = pd.merge(admissions, patients, on='subject_id', how='left')

# Merge with diagnoses on 'hadm_id'
# For simplicity, take the first diagnosis for each hospital admission.
first_diag = diagnoses.groupby('hadm_id').first().reset_index()
# Here, we use the 'icd_code' as the diagnosis.
df = pd.merge(df, first_diag[['hadm_id', 'icd_code']], on='hadm_id', how='left')
# Optionally, rename icd_code to diagnosis_code for clarity.
df = df.rename(columns={'icd_code': 'diagnosis_code'})

# Group prescriptions by 'hadm_id'
first_rx = prescriptions.groupby('hadm_id').first().reset_index()

# Merge with prescriptions on 'hadm_id'
# Use 'dose_val_rx' as dosage since 'dosage' column is not available in your data.
df = pd.merge(df, first_rx[['hadm_id', 'drug', 'dose_val_rx']], on='hadm_id', how='left')
df = df.rename(columns={'dose_val_rx': 'dosage'})

# Merge with lab events for a specific lab test (e.g., serum glucose).
# You must know the itemid for the lab test you're interested in. For demonstration, assume itemid 11289 is glucose.
glucose = labevents[labevents['itemid'] == 11289]
# Compute average glucose value per admission
glucose_avg = glucose.groupby('hadm_id')['valuenum'].mean().reset_index().rename(columns={'valuenum': 'avg_glucose'})
df = pd.merge(df, glucose_avg, on='hadm_id', how='left')

# ----------------------------
# 3. Aggregate Lab Events Data
# ----------------------------
# For example, if you have an "itemid" for blood glucose (adjust the list as needed)
glucose_item_ids = [50821, 50931]  # Example item IDs for glucose

# Filter lab events for glucose and compute average glucose per hospital admission
avg_glucose = (
    labevents[labevents['itemid'].isin(glucose_item_ids)]
    .groupby('hadm_id')['valuenum']
    .mean()
    .reset_index()
    .rename(columns={'valuenum': 'avg_glucose_value'})
)

# Merge the aggregated lab events into the main DataFrame
df = pd.merge(df, avg_glucose, on='hadm_id', how='left')

# Optionally, if you wish to map other lab results or merge additional lab event mappings,
# repeat similar steps with appropriate filters and aggregation functions.

# ----------------------------
# 4. Merge Lab Events Data (Average Glucose)
# ----------------------------
glucose_item_ids = [50821, 50931]  # Example item IDs for glucose
avg_glucose = (
    labevents[labevents['itemid'].isin(glucose_item_ids)]
    .groupby('hadm_id')['valuenum']
    .mean()
    .reset_index()
    .rename(columns={'valuenum': 'avg_glucose'})
)

df = pd.merge(df, avg_glucose, on='hadm_id', how='left')

# ----------------------------
# 3. Preprocess Data
# ----------------------------
# Function to calculate age from date of birth (assumed format "YYYY-MM-DD")
def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return None

# Compute age from dob (assuming the patients.csv has a column "dob")
df['age'] = df['anchor_age']
# Optionally, convert admittime and dischtime to datetime if needed:
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])

# Rename/assign columns for clarity
df = df.rename(columns={
    'drug': 'prescribed_medication'
})

# Drop rows with missing critical fields (e.g., age)
df = df.dropna(subset=['age'])

# ----------------------------
# 5. Rename Columns for Clarity
# ----------------------------

glucose_item_ids = [50821, 50931]  # Example item IDs for glucose
avg_glucose = (
    labevents[labevents['itemid'].isin(glucose_item_ids)]
    .groupby('hadm_id')['valuenum']
    .mean()
    .reset_index()
    .rename(columns={'valuenum': 'avg_glucose'})
)

df = pd.merge(df, avg_glucose, on='hadm_id', how='left')

# Rename prescription column to 'prescribed_medication'
df = df.rename(columns={'drug': 'prescribed_medication'})

# ----------------------------
# 6. Select Only the Required Columns and Save to CSV
# ----------------------------
final_df = df[['subject_id', 'hadm_id', 'age', 'gender', 'diagnosis_code',
               'prescribed_medication', 'dosage', 'avg_glucose']]

final_df.to_csv('d:/arogo/auto_presc_RAG/merged_data.csv', index=False)