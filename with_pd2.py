import pandas as pd

# Load the dataset
df = pd.read_excel(r"merged_output.xlsx")

# Fill missing values with 'Not Specified'
# df.fillna("Not Specified", inplace=True)

# Rank products by 'qtr_highest_sale_qty' in descending order
df["rank"] = df["qtr_highest_sale_qty"].rank(method="first", ascending=False)

# Select top 10 selling products
top_10 = df[df["rank"] <= 10]

# Select bottom 10 least-selling products
bottom_10 = df[df["rank"] > (df["rank"].max() - 10)]

# Define the feature columns to analyze
# feature_columns = ["material", "Bag Width", "Number of Interior Pockets", "Number of Exterior Pockets", "color_count"]

# Initialize lists for successful columns
successful_features = []
top_10_values = []
bottom_10_values = []

for col in df.columns:
    try:
        # Check if column exists and has valid data
        if col in df.columns and not df[col].isna().all():
            top_value = top_10[col].mode()[0] if not top_10[col].mode().empty else "Not Available"
            bottom_value = bottom_10[col].mode()[0] if not bottom_10[col].mode().empty else "Not Available"
            
            successful_features.append(col)
            top_10_values.append(top_value)
            bottom_10_values.append(bottom_value)
        else:
            print(f"Skipping column: {col} due to missing or invalid data")
    except Exception as e:
        print(f"Error processing column: {col}. Error: {e}")

# Create output DataFrame with only successful columns
output_df = pd.DataFrame({
    "Feature Name": successful_features,
    "Top 10 Value": top_10_values,
    "Bottom 10 Value": bottom_10_values
})

# Display output
print(output_df)

# Save to Excel
output_df.to_excel("common_features_top_vs_bottom2.xlsx", index=False)
