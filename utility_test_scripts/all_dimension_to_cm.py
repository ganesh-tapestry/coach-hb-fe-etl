import pandas as pd
import re

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
    length_columns = [col for col in df.columns if df[col].astype(str).str.contains(r'\d+\s*(mm|cm|in|inch|inches)', regex=True).any()]

    for col in length_columns:
        df[[f"{col}_numeric", f"{col}_unit"]] = df[col].apply(lambda x: pd.Series(extract_value_and_unit(x)))
        df[col] = df.apply(lambda row: convert_to_cm(row[f"{col}_numeric"], row[f"{col}_unit"]), axis=1)

        # Drop intermediate columns
        df.drop(columns=[f"{col}_numeric", f"{col}_unit"], inplace=True)

    return df

# Example DataFrame
data = {
    'bag_width': ['50mm', '2 inch', '100 cm', '5 inch', '75 cm'],
    'handle_length': ['30mm', '1 inch', '50 cm', '3 inch', '40 cm'],
    'random_col': ['hello', 'world', 'text', 'no units', 'skip this']
}

df = pd.DataFrame(data)

# Convert all length columns
df = convert_length_columns_to_cm(df)

print(df)
