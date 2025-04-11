import pandas as pd

# Load the Excel and CSV files
features_df = pd.read_excel(r"cleaned_bag_features_data.xlsx")  # style_id
df2 = pd.read_csv(r"input_data/Sales_product_data_postgres_dump_19-03-2025.csv")  # style_code

# Rename columns for consistency
features_df.rename(columns={"style_id": "style"}, inplace=True)
# df2.rename(columns={"style_code": "style"}, inplace=True)


print(len(list(set(features_df["style"]) & set(df2["style"]))))

# Merge DataFrames on 'style'
merged_df = features_df.merge(df2, on="style", how="inner")

# Save the merged result to a new Excel file
merged_df.to_excel("HBA_SALES_DATA_v1.xlsx", index=False)

print(merged_df.shape)

