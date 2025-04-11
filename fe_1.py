# Model: I'll help you prepare a Python code to handle the data issues in this Excel file. Based on my analysis, the data contains many missing values (represented as "Not specified", "None", "NA", etc.), categorical variables, and numerical variables that need cleaning. Let's create a comprehensive data preparation pipeline.

# ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import re

# Load the data
# file_path = 'feature_table_data_generator_output1 - Copy.xlsx'
# file_path= r"merged_output.xlsx"
# df = pd.read_excel(file_path)

# # Load three Excel files
feature_table_df = pd.read_excel(r"input_data\feature_table_data_generator_output.xlsx")
product_df = pd.read_excel(r"input_data\final_encoded_products 3.xlsx")
sales_df = pd.read_excel(r"input_data\sales_data_product_style_codes_grouped_sum_C_series_part2 (1).xlsx")

for i in  feature_table_df:
    if "material_type" in i:
        print("feature_table_df",i)


for i in  product_df:
    if "material_type" in i:
        print("product_df",i)
        
for i in  sales_df:
    if "material_type" in i:
        print("sales_df",i)           
             
# Rename columns for consistency
feature_table_df.rename(columns={"style_id": "style"}, inplace=True)
product_df.rename(columns={"style_code": "style"}, inplace=True)
# sales_df already has 'style', so no renaming needed

# Merge DataFrames on 'style'
merged_df = feature_table_df.merge(product_df, on="style", how="inner").merge(sales_df, on="style", how="inner")

df = merged_df

# Find duplicate column names
duplicate_columns = df.columns[df.columns.duplicated()].tolist()

print("Duplicate column names:", duplicate_columns)



def print_function_name_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"{'*' * 50} {func.__name__} {'*' * 50}")
        return func(*args, **kwargs)
    return wrapper

# Function to check and print initial data information
def check_data_info(df):
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns with missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\nSample of the data:")
    print(df.head())

# Check initial data information
check_data_info(df)

# Function to clean column names
def clean_column_names(df):
    # Remove leading/trailing spaces and convert to lowercase
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

# Function to identify and handle missing values
def handle_missing_values(df):
    # Identify variations of missing values in text form
    missing_values_variants = ["Not specified", "None", "NA", "Not applicable", 
                              "N/A", "Absent", "No", "No handle", "None", ""]
    
    # Replace all variants of missing values with np.nan
    for variant in missing_values_variants:
        df = df.replace(variant, np.nan)
    
    # Calculate percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100
    
    # Columns with high missing values (>50%) might be dropped
    high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
    print(f"\nColumns with >50% missing values (consider dropping): {high_missing_cols}")
    
    print(" Dropping Columns: ", high_missing_cols)
    df.drop(columns = high_missing_cols, inplace=True)
    # For now, we'll keep all columns but handle them differently based on data type
    return df

# Function to extract numeric values from measurements
def extract_numeric_values(value):
    if pd.isnull(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return value
    
    # Try to extract numeric part using regex
    if isinstance(value, str):
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', value)
        if matches:
            return float(matches[0])
    
    return np.nan


def convert_mm_to_cm(value, unit):
    if unit == 'mm':
        return value / 10
    return value

# # Function to process dimensional features
# def process_dimensional_features(df):
#     # List of dimensional features
#     dimensional_features = ['bag_height', 'bag_width', 'gusset_width', 'handle_drop_length',
#                            'long_strap_drop_length', 'thickness', "top_opening_width"]
    
#     # Extract numeric values
#     for col in dimensional_features:
#         if col in df.columns:
#             df[col] = df[col].apply(extract_numeric_values)
            
#             # Add unit column if needed
#             df[f'{col}_unit'] = 'cm'  # Default unit is cm
            
#             # Check for 'inches' in the original column
#             if col in df.columns:
#                 df[f'{col}_unit'] = df[col].astype(str).apply(
#                     lambda x: 'inches' if 'inch' in str(x).lower() else 'cm'
#                 )
                
#                 # Convert inches to cm for consistency
#                 inches_mask = df[f'{col}_unit'] == 'inches'
#                 # 1 inch = 2.54cm
#                 df.loc[inches_mask, col] = df.loc[inches_mask, col] * 2.54
#                 df.loc[inches_mask, f'{col}_unit'] = 'cm'
                
#                 # Convert mm to cm
#                 df[col] = df.apply(lambda row: convert_mm_to_cm(row[col], row[f'{col}_unit']), axis=1)
#                 df[f'{col}_unit'] = 'cm'                
    
#     return df


def convert_length_columns_to_cm(df):
    """
    Identifies columns with length values containing units (mm, cm, inch) 
    and converts them all to cm.
    """
    def extract_value_and_unit(value):
        """Extracts numeric value and unit from a string."""
        match = re.match(r"([\d\.]+)\s*(mm|cm|in|inch|inches)?", str(value).lower())
        if match:
            num_value = float(match.group(1))  # Extract numeric part
            unit = match.group(2) if match.group(2) else 'cm'  # Default to cm if no unit found
            unit = 'inch' if unit in ['in', 'inch', 'inches'] else unit  # Normalize inch variations
            return num_value, unit
        return None, None

    def convert_to_cm(value, unit):
        """Converts mm and inches to cm."""
        if unit == 'mm':
            return value / 10
        elif unit == 'inch':
            return value * 2.54
        elif unit == 'cm':
            return value
        return None  # Handle invalid cases gracefully

    # Identify columns with potential dimensional data
    # length_columns = [col for col in df.columns if df[col].astype(str).str.contains(r'\d+\s*(mm|cm|in|inch|inches)', regex=True).any()]
    pattern = re.compile(r'(?i)\b(length|height|width|depth|diameter|size|dimension)\b')
    
    dimensional_columns = [col for col in df.columns if pattern.search(col)]

    for col in dimensional_columns:
        df[[f"{col}_numeric", f"{col}_unit"]] = df[col].apply(lambda x: pd.Series(extract_value_and_unit(x)))
        df[col] = df.apply(lambda row: convert_to_cm(row[f"{col}_numeric"], row[f"{col}_unit"]), axis=1)

        # Drop intermediate columns
        df.drop(columns=[f"{col}_numeric", f"{col}_unit"], inplace=True)

    return df


# Function to encode categorical variables
def encode_categorical_features(df):
    categorical_features = [
        'material_type', 'leather_texture', 'hardware_color', 'closure_type',
        'logo_visibility', 'silhouette_type', 'color', 'interior_lining_material',
        'strap_type', 'edge_finishing', 'hardware_quality', 'embellishment_type',
        'bottom_structure', 'collection_line', 'seasonal_relevance'
    ]
    
    # Create label encoders for each categorical feature
    encoders = {}
    
    for col in categorical_features:
        if col in df.columns:
            # Fill missing values with 'Unknown' before encoding
            df[col] = df[col].fillna('Unknown')
            
            # Create a label encoder
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col]) # issue here
            
            # Store the encoder for future reference
            encoders[col] = le
            
            # Create dummy variables for categorical features with few unique values
            if df[col].nunique() < 10:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
    
    return df, encoders

# Function to handle numerical features
def handle_numerical_features(df):
    # List of numerical features
    numerical_features = [
        'number_of_compartments', 'number_of_interior_pockets', 'number_of_exterior_pockets',
        'top_opening_width', 'card_slot_count'
    ]
    
    for col in numerical_features:
        if col in df.columns:
            # Convert to numeric, errors='coerce' will convert non-numeric values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values with median
            # median_value = df[col].median()
            # df[col] = df[col].fillna(median_value)
            df[col].fillna(0, inplace=True)
    
    return df

# Function to handle carry options (these are binary features)
def handle_carry_options(df):
    carry_options = [
        'crossbody_option', 'shoulder_option', 'hand_carry_option', 'arm_carry_option',
        'backpack_option', 'belt_bag_option', 'wristlet_option', 'clutch_option'
    ]
    
    for col in carry_options:
        if col in df.columns:
            # Map 'Yes' to 1 and 'No' to 0
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            df[col] = df[col].fillna(0)  # Assume missing means not available
    
    return df

# Function to create aggregated features
def create_aggregated_features(df):
    # Count number of carry options
    carry_options = [col for col in df.columns if '_option' in col]
    if carry_options:
        df['total_carry_options'] = df[carry_options].sum(axis=1)
    
    # Count number of pockets
    pocket_cols = [col for col in df.columns if 'pocket' in col]
    if pocket_cols:
        df['total_pockets'] = df[pocket_cols].sum(axis=1)
    
    # Structured vs. Slouchy as numeric
    if 'structured_vs._slouchy' in df.columns:
        structure_map = {
            'Structured': 1,
            'Flat': 0.75,
            'Semi-structured': 0.5,
            'Soft': 0.25,
            'Slouchy': 0
        }
        df['structure_score'] = df['structured_vs._slouchy'].map(structure_map)
        df['structure_score'] = df['structure_score'].fillna(0.5)  # Default to middle value
        df.drop(columns = ['structured_vs._slouchy'], inplace=True)
    return df



# Function to handle binary features
def handle_binary_features(df):
    # List of binary features
    # binary_features = [
    #     'limited_edition_status', 'signature_pattern_presence', 'colorblock_design',
    #     'quilted_pattern', 'stitch_visibility', 'corner_protection', 'bottom_feet',
    #     'hangtag_presence', 'chain_detail', 'turnlock_hardware', 'removable_strap',
    #     'adjustable_strap', 'laptop_compatibility', 'blind_emboss_detail',
    #     'foil_emboss_detail', 'contrast_stitching', 'piping_detail', 'bombe_edge',
    #     'tapered_shape', 'reinforced_handle_attachment', 'storypatch_presence',
    #     'collapsible_design', 'dual_handle_design', 'wristlet_attachment',
    #     'keyring/key_leash', 'gusseted_pocket', 'zipper_extension', 'dust_bag_included',"coin_pocket",
    #         "water_bottle_pocket", "exterior_slip_pocket", "interior_zip_pocket", "interior_zip_pocket", "interior_slip_pocket","exterior_zip_pocket"
    # ]

    positive_values = ['Yes', "yes", 'Present', "present", 'True',"true"]
    negative_values = ['No', "no", 'Absent', "absent", 'False', "false", str(np.nan) ,str(None)]
    binary_features = list()
    
    for column in df.columns:
        if df[column].isin(['yes', 'no',"Yes","No"]).any():
            binary_features.append(column)
    
    print (" all binary_features: ", binary_features)
    # Convert binary features to 0/1
    for col in binary_features:
        if col in df.columns:
            # Map 'Yes', 'Present', etc. to 1 and 'No', 'Absent', etc. to 0
            positive_values = ['Yes', "yes", 'Present', "present", 'True',"true"]
            negative_values = ['No', "no", 'Absent', "absent", 'False', "false", str(np.nan) ,str(None)]
            
            # Create a mapping function
            def map_binary(val):
                if pd.isnull(val):
                    # return np.nan
                    return 0
                elif any(pos in str(val) for pos in positive_values):
                    return 1
                elif any(neg in str(val) for neg in negative_values):
                    return 0
                else:
                    return 0
                    # return np.nan
            
            df[col] = df[col].apply(map_binary)

    return df


# # Function to convert yes/no columns to binary
# @print_function_name_decorator
# def convert_yes_no_to_binary(df):
#     for column in df.columns:
#         # print(df[column].value_counts())

#         if df[column].isin(['yes', 'no',"Yes","No"]).any():
#             print(column, "is a binary column")
#             print(df[column].value_counts())
#             df[column] = df[column].map({'yes': 1, 'no': 0})
#     return df


# Main data preparation pipeline
def prepare_data_for_analysis(df):
    # Clean column names
    df = clean_column_names(df)


    # Process dimensional features
    # df = process_dimensional_features(df)
    df = convert_length_columns_to_cm(df)
    print(1,df["tablet_compatibility"].value_counts())

    # Handle binary features
    df = handle_binary_features(df)
    print(2,df["tablet_compatibility"].value_counts())
    
    # Handle numerical features
    df = handle_numerical_features(df)
    print(3,df["tablet_compatibility"].value_counts())
    
    # Handle carry options
    df = handle_carry_options(df)
    print(4,df["tablet_compatibility"].value_counts())
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    print(5,df["tablet_compatibility"].value_counts())
    
    # Create aggregated features
    df = create_aggregated_features(df)
    print(6,df["tablet_compatibility"].value_counts())
    
    


    # # Convert yes/no columns to binary
    # df = convert_yes_no_to_binary(df)

    
    
    # Fill remaining NaN values
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    
    
    # For categorical columns, fill with mode
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    categorical_cols.remove("style")
    categorical_cols.remove("product_name")

    # Handle missing values
    df = handle_missing_values(df)

    #TODO : Issue -  This is causing lot of incorrect values 
    # for col in categorical_cols:
    #     df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

# Apply the data preparation pipeline
df_cleaned = prepare_data_for_analysis(df)

print("main",df["tablet_compatibility"])

# Check the processed data
print("\nCleaned data information:")
print(f"Shape: {df_cleaned.shape}")
print("\nSample of cleaned data:")
print(df_cleaned.head())

# Check for any remaining missing values
missing_after = df_cleaned.isnull().sum()
print("\nRemaining missing values:")
print(missing_after[missing_after > 0])

# Save the cleaned data to a new file
df_cleaned.to_excel('cleaned_bag_features_data.xlsx', index=False)
print("\nCleaned data saved to 'cleaned_bag_features_data.xlsx'")

# Generate basic statistics for numerical columns
numeric_stats = df_cleaned.describe()
print("\nStatistics for numerical columns:")
print(numeric_stats)

# # Create a correlation matrix for numerical features
# def plot_correlation_matrix(df):
#     # Select only numeric columns
#     numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
#     # Calculate correlation matrix
#     corr_matrix = numeric_df.corr()
    
#     # Plot correlation matrix
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title('Correlation Matrix of Numerical Features')
#     plt.tight_layout()
#     plt.savefig('correlation_matrix.png')
#     plt.close()
    
#     # Return top correlations
#     corr_pairs = []
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if abs(corr_matrix.iloc[i, j]) > 0.5:  # Only strong correlations
#                 corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
#     # Sort by absolute correlation value
#     corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
#     return corr_pairs

# # Plot correlation matrix and get top correlations
# top_correlations = plot_correlation_matrix(df_cleaned)
# print("\nTop feature correlations:")
# for feature1, feature2, corr_value in top_correlations[:10]:
#     print(f"{feature1} and {feature2}: {corr_value:.3f}")

# # Function to visualize the distribution of key features
# def plot_key_features_distribution(df):
#     # Select some key numerical features
#     key_numeric_features = ['bag_height', 'bag_width', 'gusset_width', 'number_of_compartments', 
#                            'number_of_interior_pockets', 'structure_score']
    
#     # Filter out features that don't exist in the dataframe
#     key_numeric_features = [col for col in key_numeric_features if col in df.columns]
    
#     # Plot histograms
#     fig, axes = plt.subplots(len(key_numeric_features), 1, figsize=(10, 3*len(key_numeric_features)))
    
#     for i, feature in enumerate(key_numeric_features):
#         sns.histplot(df[feature].dropna(), ax=axes[i])
#         axes[i].set_title(f'Distribution of {feature}')
#         axes[i].set_xlabel(feature)
#         axes[i].set_ylabel('Count')
    
#     plt.tight_layout()
#     plt.savefig('key_features_distribution.png')
#     plt.close()

# # Plot key features distribution
# # plot_key_features_distribution(df_cleaned)
# # print("\nKey features distribution plots saved to 'key_features_distribution.png'")

# # Function to visualize categorical features
# def plot_categorical_features(df):
#     # Select some key categorical features
#     key_cat_features = ['silhouette_type', 'material_type', 'closure_type', 'seasonal_relevance']
    
#     # Filter out features that don't exist in the dataframe
#     key_cat_features = [col for col in key_cat_features if col in df.columns]
    
#     # Plot bar charts
#     fig, axes = plt.subplots(len(key_cat_features), 1, figsize=(12, 4*len(key_cat_features)))
    
#     for i, feature in enumerate(key_cat_features):
#         value_counts = df[feature].value_counts().sort_values(ascending=False).head(10)
#         sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
#         axes[i].set_title(f'Top 10 values for {feature}')
#         axes[i].set_xlabel(feature)
#         axes[i].set_ylabel('Count')
#         axes[i].tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     plt.savefig('categorical_features_distribution.png')
#     plt.close()

# # Plot categorical features
# # plot_categorical_features(df_cleaned)
# # print("\nCategorical features distribution plots saved to 'categorical_features_distribution.png'")

# print("\nData preparation complete. The dataset is now ready for analysis.")
# # ```

# This comprehensive code includes the following data preparation steps:

# 1. **Loading and Initial Inspection**: Loads the Excel file and provides initial information about the dataset.

# 2. **Cleaning Column Names**: Standardizes column names by removing spaces, converting to lowercase, and replacing spaces with underscores.

# 3. **Handling Missing Values**: Identifies various forms of missing values (like "Not specified", "None", etc.) and converts them to NaN for consistent handling.

# 4. **Processing Dimensional Features**: Extracts numeric values from measurements like bag height, width, etc., and standardizes units (converting inches to cm).

# 5. **Encoding Categorical Features**: Uses label encoding for categorical variables and creates dummy variables for those with few unique values.

# 6. **Handling Binary Features**: Converts binary features (like "Yes"/"No", "Present"/"Absent") to 0/1 values.

# 7. **Handling Numerical Features**: Ensures numerical features are properly formatted and fills missing values with medians.

# 8. **Handling Carry Options**: Processes the various carry option columns as binary features.

# 9. **Creating Aggregated Features**: Derives new features like total carry options, total pockets, and structure score.

# 10. **Final Cleanup**: Fills any remaining missing values with appropriate strategies.

# 11. **Data Visualization**: Creates correlation matrices and feature distribution plots to help understand the data.

# The code saves the cleaned dataset to a CSV file and generates visualizations to help understand the relationships between features. This processed data is now ready for further analysis, including building predictive models for sales.