import pandas as pd

# # Sample DataFrame
# data = {
#     "style": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"],
#     "qty": [500, 450, 400, 350, 300, 275, 260, 250, 240, 230, 220],  # Sales quantity
#     "material": ["Leather", "Canvas", "Leather", "Leather", "Canvas", "Leather", "Canvas", "Leather", "Leather", "Canvas", "Leather"],
#     "width": [12, 10, 12, 14, 10, 12, 10, 12, 14, 10, 12],
#     "num_pockets": [3, 2, 3, 4, 2, 3, 2, 3, 4, 2, 3],
#     "strap_option_count": [2, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2],
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

import pandas as pd

# # Load three Excel files
# df1 = pd.read_excel(r"input_data\feature_table_data_generator_output.xlsx")
# df2 = pd.read_excel(r"input_data\final_encoded_products 3.xlsx")
# df3 = pd.read_excel(r"input_data\sales_data_product_style_codes_grouped_sum_C_series_part2 (1).xlsx")


# # Rename columns for consistency
# df1.rename(columns={"style_id": "style"}, inplace=True)
# df2.rename(columns={"style_code": "style"}, inplace=True)
# # df3 already has 'style', so no renaming needed

# # Merge DataFrames on 'style'
# merged_df = df1.merge(df2, on="style", how="outer").merge(df3, on="style", how="outer")


# # Merge DataFrames on 'style' and 'style_id'
# # merged_df = df1.merge(df2, on=["style_id", "style_code"], how="outer").merge(df3, on=["style_id", "style"], how="outer")

# # Save the merged result to a new Excel file
# merged_df.to_excel("merged_output.xlsx", index=False)


#############################################


df = pd.read_excel(r"merged_output.xlsx")

# Rank products by 'qty' in descending order
df["rank"] = df["qtr_highest_sale_qty"].rank(method="first", ascending=False)

# df.fillna("Not Specified", inplace=True)


# Select top 10 products
top_10 = df[df["rank"] <= 10]

# Find common features in the top 10 products
common_features = {}
# for col in ["material", "Bag Width", "Number of Interior Pockets", "Number of Exterior Pockets", "color_count","Divided Interior"]:
for col in df.columns:
    try:
        common_features[col] = top_10[col].mode()[0]  # Most frequent value
    except Exception as e:
        print(e)
        pass
# # Display results
# print("Common Features in Top 10 Selling Products:")
# for feature, value in common_features.items():
#     print(f"{feature}: {value}")

############### bottom 10 ####################################


# # Rank products by 'qtr_highest_sale_qty' in descending order
# df["rank"] = df["qtr_highest_sale_qty"].rank(method="first", ascending=False)

# Select top 10 selling products
# top_10 = df[df["rank"] <= 10]

# Select bottom 10 least-selling products
bottom_10 = df[df["rank"] > (df["rank"].max() - 10)]

# Define the feature columns to analyze
# feature_columns = ["material", "Bag Width", "Number of Interior Pockets", "Number of Exterior Pockets", "color_count"]

# Find common features in top 10 products
common_features_top_10 = {col: top_10[col].mode()[0] for col in feature_columns}

# Find common features in bottom 10 products
common_features_bottom_10 = {col: bottom_10[col].mode()[0] for col in feature_columns}

# Display results
print("Common Features in Top 10 Selling Products:")
for feature, value in common_features_top_10.items():
    print(f"{feature}: {value}")

print("\nCommon Features in Bottom 10 Selling Products:")
for feature, value in common_features_bottom_10.items():
    print(f"{feature}: {value}")
    

# Create output DataFrame
output_df = pd.DataFrame({
    "Feature Name": feature_columns,
    "Top 10 Value": [common_features_top_10[col] for col in feature_columns],
    "Bottom 10 Value": [common_features_bottom_10[col] for col in feature_columns]
})

# Display output
print(output_df)

# Save to Excel
output_df.to_excel("common_features_top_vs_bottom.xlsx", index=False)    