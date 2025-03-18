
import pandas as pd

# # Load three Excel files
df1 = pd.read_excel(r"input_data\feature_table_data_generator_output.xlsx")
df2 = pd.read_excel(r"input_data\final_encoded_products 3.xlsx")
df3 = pd.read_excel(r"input_data\sales_data_product_style_codes_grouped_sum_C_series_part2 (1).xlsx")


# Get the column names for each DataFrame
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)
columns_df3 = set(df3.columns)


if "material_type" in columns_df1:
    print("df1")
    
if "material_type" in columns_df3:
    print("df2")
    
if "material_type" in columns_df3:
    print("df3")        
    
# # Find the common column names
# common_columns = columns_df1.intersection(columns_df2)#.intersection(columns_df3)

# print("Common column names:", common_columns)