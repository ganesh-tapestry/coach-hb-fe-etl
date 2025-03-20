# # Number of files matching style codes: 2310

# import os
# import pandas as pd

# def get_all_filenames(folder_paths):
#     """Retrieve all file names from the given folder paths, including subdirectories."""
#     file_names = []
#     for folder in folder_paths:
#         if os.path.exists(folder):
#             for root, _, files in os.walk(folder):  # Recursively get all files
#                 file_names.extend([file.lower() for file in files])  # Convert to lowercase
#     return file_names

# def count_matching_files(style_codes, file_names):
#     """Count how many file names contain any unique style codes (case insensitive)."""
#     style_codes = set(style_codes.dropna().astype(str).str.lower())  # Ensure unique, lowercase style codes
#     match_count = sum(any(style_code in file_name for style_code in style_codes) for file_name in file_names)
#     return match_count

# # Load CSV with style codes
# csv_df = pd.read_csv(r"input_data\style_codes_W_WHB_only38k.csv")  # Replace with actual CSV file
# style_codes = csv_df['style_code']  # Extract style codes

# # Define folder paths
# folder_paths = [r"D:\Users\gkharad\Downloads\OUTLET (1)", r"D:\Users\gkharad\Downloads\Outlet"]  # Replace with actual folder paths

# # Get all file names (case insensitive, recursive)
# file_names = get_all_filenames(folder_paths)

# # Count matching files
# matching_count = count_matching_files(style_codes, file_names)

# print(f"Number of files matching style codes: {matching_count}")


import os
import pandas as pd

# Define folder paths
folder_paths = [r"D:\Users\gkharad\Downloads\OUTLET (1)", r"D:\Users\gkharad\Downloads\Outlet"]  # Replace with actual folder paths

# Read CSV containing style codes
csv_df = pd.read_csv(r"input_data\POC_Target_style_id.csv")  # Update with actual CSV file path
style_codes = set(csv_df['style_code'].dropna().astype(str).str.lower())  # Convert to lowercase and ensure unique values

# Dictionary to store matches {style_code: [file_paths]}
matches = {}

# Traverse both folder paths including subdirectories
for folder in folder_paths:
    for root, _, files in os.walk(folder):
        for file in files:
            file_lower = file.lower()  # Convert file name to lowercase
            for style_code in style_codes:
                if style_code in file_lower:
                    print(style_code, file)
                    full_path = os.path.join(root, file)  # Get full file path
                    if style_code in matches:
                        matches[style_code].append(full_path)
                    else:
                        matches[style_code] = [full_path]

# Convert match dictionary to DataFrame
match_list = []
for style_code, file_paths in matches.items():
    for file_path in file_paths:
        match_list.append({"style_code": style_code, "file_path": file_path})

# Save results to Excel
match_df = pd.DataFrame(match_list)
output_file = "style_code_file_matches_target.xlsx"
match_df.to_excel(output_file, index=False)


print("Unique style ids are: ",match_df["style_code"].nunique())
print(f"Matching results saved in {output_file}")


