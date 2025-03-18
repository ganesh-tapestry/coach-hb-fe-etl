
import pandas as pd

# # Load three Excel files
df1 = pd.read_excel(r"input_data\feature_table_data_generator_output.xlsx")
df2 = pd.read_excel(r"input_data\final_encoded_products 3.xlsx")
df3 = pd.read_excel(r"input_data\sales_data_product_style_codes_grouped_sum_C_series_part2 (1).xlsx")


# Rename columns for consistency
df1.rename(columns={"style_id": "style"}, inplace=True)
df2.rename(columns={"style_code": "style"}, inplace=True)
# df3 already has 'style', so no renaming needed

# Merge DataFrames on 'style'
merged_df = df1.merge(df2, on="style", how="inner").merge(df3, on="style", how="inner")


# Merge DataFrames on 'style' and 'style_id'
# merged_df = df1.merge(df2, on=["style_id", "style_code"], how="outer").merge(df3, on=["style_id", "style"], how="outer")

# Save the merged result to a new Excel file
merged_df.to_excel("lft_joined_merged_output.xlsx", index=False)

print(merged_df.shape)