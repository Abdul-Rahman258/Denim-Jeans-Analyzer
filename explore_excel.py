import pandas as pd
import glob

excel_files = glob.glob(r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\Data\*.xlsx')
for f in excel_files:
    print(f"--- {f} ---")
    try:
        df = pd.read_excel(f)
        print("Columns:", df.columns.tolist())
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {f}: {e}")
