import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel(r"input_data\feature_table_data_generator_output.xlsx")



# -------------------- HANDLE DUPLICATE COLUMNS --------------------
def rename_duplicate_columns(df):
    """Renames duplicate column names by appending a unique index."""
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df

df = rename_duplicate_columns(df)

# -------------------- CLEAN COLUMN NAMES --------------------
def clean_column_names(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

df = clean_column_names(df)



# -------------------- HANDLE MISSING VALUES --------------------
def handle_missing_values(df,drop_threshold = 50):
    """Handles missing values by replacing known missing indicators and dropping high-missing columns."""
    missing_values_variants = ["Not specified", "None", "NA", "Not applicable", "N/A", "Absent", "", str(np.nan)]
    df.replace(missing_values_variants, np.nan, inplace=True)

    # Drop columns with >50% missing values
    missing_percentage = df.isnull().mean() * 100
    high_missing_cols = missing_percentage[missing_percentage > drop_threshold].index.tolist()
    print("high_missing_cols : ", high_missing_cols)
    df.drop(columns=high_missing_cols, inplace=True)

    return df

df = handle_missing_values(df, drop_threshold=90)

# -------------------- HANDLE BINARY FEATURES --------------------
def handle_binary_features(df):
    binary_features = [col for col in df.columns if df[col].astype(str).str.lower().isin(['yes', 'no', 'present', 'true', "available", 'false']).any()]

    def map_binary(val):
        if pd.isnull(val): return 0
        if str(val).lower() in ['yes', 'present', 'true', "available"]: return 1
        return 0

    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].apply(map_binary)

    return df

df = handle_binary_features(df)



# -------------------- CONVERT LENGTH UNITS TO CM --------------------
def convert_length_columns_to_cm(df):
    """Identifies and converts length columns from mm, inch to cm."""
    def extract_value_and_unit(value):
        match = re.match(r"([\d\.]+)\s*(mm|cm|in|inch|inches)?", str(value).lower())
        if match:
            num_value = float(match.group(1))
            unit = match.group(2) if match.group(2) else 'cm'
            unit = 'inch' if unit in ['in', 'inch', 'inches'] else unit
            return num_value, unit
        return None, None

    def convert_to_cm(value, unit):
        if unit == 'mm': return value / 10
        if unit == 'inch': return value * 2.54
        return value  # Default to cm

    # length_columns = [col for col in df.columns if df[col].astype(str).str.contains(r'\d+\s*(mm|cm|in|inch|inches)', regex=True).any()]
    pattern = re.compile(r'(?i)(length|height|width|depth|diameter|size|dimension)')
    
    dimensional_columns = [col for col in df.columns if pattern.search(col)]    
    dimensional_columns = [col for col in df.columns if pattern.search(col)]
    print("length_columns are :", dimensional_columns)
    for col in dimensional_columns:
        df[[f"{col}_numeric", f"{col}_unit"]] = df[col].apply(lambda x: pd.Series(extract_value_and_unit(x)))
        df[col] = df.apply(lambda row: convert_to_cm(row[f"{col}_numeric"], row[f"{col}_unit"]), axis=1)
        df.drop(columns=[f"{col}_numeric", f"{col}_unit"], inplace=True)
    
    return df

df = convert_length_columns_to_cm(df)

# -------------------- HANDLE NUMERICAL FEATURES --------------------
def handle_numerical_features(df):
    numerical_features = [
        'number_of_compartments', 'number_of_interior_pockets', 'number_of_exterior_pockets',
        'top_opening_width', 'card_slot_count'
    ]
    
    for col in numerical_features:
        if col in df.columns:
            # Convert numeric-like strings to numbers, replace non-convertible text with 0
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace NaNs (including text values that were converted) with median or 0 if median is NaN
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value if not pd.isna(median_value) else 0)
    
    return df
df = handle_numerical_features(df)

# -------------------- HANDLE CATEGORICAL FEATURES --------------------
def encode_categorical_features(df):
    categorical_features = [
        'material_type', 'leather_texture', 'hardware_color', 'closure_type',
        'logo_visibility', 'silhouette_type', 'color', 'interior_lining_material',
        'strap_type', 'edge_finishing', 'hardware_quality', 'embellishment_type',
        'bottom_structure', 'collection_line', 'seasonal_relevance'
    ]
    
    encoders = {}
    for col in categorical_features:
        if col in df.columns:
            print("encode_categorical_features for col name: ", col)
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders

df, encoders = encode_categorical_features(df)

import pickle

# Save encoders for future use
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
    
# # Load encoders
# with open('label_encoders.pkl', 'rb') as f:
#     loaded_encoders = pickle.load(f)    

for col, le in encoders.items():
    print(f"\nMapping for {col}:")
    for class_label, encoded_value in zip(le.classes_, le.transform(le.classes_)):
        print(f"{encoded_value} → {class_label}")

# -------------------- HANDLE CARRY OPTIONS --------------------
def handle_carry_options(df):
    carry_options = [col for col in df.columns if '_option' in col]
    for col in carry_options:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)
    return df

df = handle_carry_options(df)

# -------------------- CREATE AGGREGATED FEATURES --------------------
def create_aggregated_features(df):
    carry_options = [col for col in df.columns if '_option' in col]
    if carry_options:
        df['total_carry_options'] = df[carry_options].sum(axis=1)
    
    pocket_cols = [col for col in df.columns if 'pocket' in col]
    if pocket_cols:
        df['total_pockets'] = df[pocket_cols].sum(axis=1)
    
    structure_map = {'Structured': 1, 'Flat': 0.75, 'Semi-structured': 0.5, 'Soft': 0.25, 'Slouchy': 0}
    if 'structured_vs._slouchy' in df.columns:
        df['structure_score'] = df['structured_vs._slouchy'].map(structure_map).fillna(0.5)
        df.drop(columns=['structured_vs._slouchy'], inplace=True)

    return df

df = create_aggregated_features(df)


# -------------------- HANDLE MISSING VALUES --------------------


df = handle_missing_values(df)

# # -------------------- FINAL CLEANUP --------------------
# def final_cleanup(df):
#     for col in df.select_dtypes(include=['object']).columns:
#         if col not in ['style', 'product_name']:
#             df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
#     return df

# df = final_cleanup(df)

# -------------------- SAVE CLEANED DATA --------------------
df.to_excel('cleaned_bag_features_data.xlsx', index=False)
print("✅ Cleaned data saved to 'cleaned_bag_features_data.xlsx'.")

# -------------------- SUMMARY --------------------
print(f"\nShape of cleaned data: {df.shape}")
print("\nSample of cleaned data:")
print(df.head())

# Check for any remaining missing values
missing_after = df.isnull().sum()
print("\nRemaining missing values:")
print(missing_after[missing_after > 0])
