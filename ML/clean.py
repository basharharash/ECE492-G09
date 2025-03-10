import os
import pandas as pd

def merge_roast_data(excel_file):
    """
    Reads two sheets from the XLS file:
      - 'Curve - Bean temperature'
      - 'Curve - Gas'
    For the "Curve - Bean temperature" sheet, renames:
       'Time (s)' to 'Time'
       'Value (CELSIUS)' to 'bean_temp'
    For the "Curve - Gas" sheet, renames:
       'Time (s)' to 'Time'
       'Value' to 'gas_setting'
    Merges them on 'Time' and returns a DataFrame with columns:
      ['Time', 'bean_temp', 'gas_setting']
    """
    # Read the two sheets
    df_temp = pd.read_excel(excel_file, sheet_name="Curve - Bean temperature")
    df_gas = pd.read_excel(excel_file, sheet_name="Curve - Gas")

    # Rename columns for the bean temperature sheet
    df_temp.rename(columns={"Time (s)": "Time", "Value (CELSIUS)": "bean_temp"}, inplace=True)
    
    # Rename columns for the gas sheet (update the gas value name if needed)
    df_gas.rename(columns={"Time (s)": "Time", "Value (PERCENT)": "gas_setting"}, inplace=True)

    # Merge on the new 'Time' column
    merged = pd.merge(df_temp, df_gas, on="Time", how="outer")
    merged.sort_values(by="Time", inplace=True)

    # Interpolate to fill any missing values
    merged.interpolate(method='linear', inplace=True)

    return merged

def process_all_files():
    # The top-level raw folders
    quarters = ["Q1", "Q2", "Q3"]
    categories = ["Valid", "Invalid"]

    for q in quarters:
        for cat in categories:
            input_dir = os.path.join("raw", q, cat)
            output_dir = os.path.join("modified", q, cat)

            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Go through each file in the input directory
            for file_name in os.listdir(input_dir):
                # Process only Excel files
                if file_name.lower().endswith((".xls", ".xlsx")):
                    full_path = os.path.join(input_dir, file_name)
                    print(f"Processing: {full_path}")

                    # Merge the data
                    try:
                        merged_df = merge_roast_data(full_path)
                    except Exception as e:
                        print(f"Error merging {file_name}: {e}")
                        continue

                    # Filter out rows where Time < 0
                    merged_df = merged_df[merged_df["Time"] >= 0]

                    # Save as CSV (same name but .csv extension)
                    csv_name = os.path.splitext(file_name)[0] + ".csv"
                    output_path = os.path.join(output_dir, csv_name)
                    merged_df.to_csv(output_path, index=False)

                    print(f"Saved merged file to: {output_path}")

if __name__ == "__main__":
    process_all_files()
