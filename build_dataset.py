import pandas as pd
import glob
import os
import re
import shutil
import numpy as np

DATA_DIR = r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\Data'
OUTPUT_DIR = r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\ProcessedData'

def clean_excel_img_name(name):
    """ Cleans 'IMG_ 31.jpg' to '31' or 'IMG_ (1).jpeg' to '1' """
    name = str(name).strip()
    match = re.search(r'(\d+)', name)
    if match:
        return match.group(1)
    return None

def clean_dir_img_name(name):
    """ Cleans 'IMG_1.jpg(1).jpeg' to '1' or 'IMG_102.jpg.jpeg' to '102' """
    name = str(name).strip()
    match = re.search(r'IMG_(\d+)', name)
    if match:
        return match.group(1)
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    excel_files = glob.glob(os.path.join(DATA_DIR, '*.xlsx'))
    image_files = glob.glob(os.path.join(DATA_DIR, '*.jpeg'))
    
    # Map cleaned numeric ID to absolute image path
    img_id_to_path = {}
    for f in image_files:
        basename = os.path.basename(f)
        idx = clean_dir_img_name(basename)
        if idx:
            img_id_to_path[idx] = f
            
    print(f"Total images found on disk: {len(image_files)}, Parseable: {len(img_id_to_path)}")
    
    all_dfs = []
    
    for f in excel_files:
        try:
            # First try reading row by row to find the header
            df_raw = pd.read_excel(f, header=None)
            header_idx = None
            for idx, row in df_raw.iterrows():
                if any(isinstance(val, str) and 'Images Name' in val for val in row.values):
                    header_idx = idx
                    break
                    
            if header_idx is not None:
                df = pd.read_excel(f, header=header_idx)
                
                # Find image col
                img_col = [c for c in df.columns if 'Images Name' in str(c)][0]
                df = df.dropna(subset=[img_col])
                df = df[df[img_col].astype(str).str.contains('IMG', na=False)]
                
                # Get ID
                df['img_id'] = df[img_col].apply(clean_excel_img_name)
                df = df.dropna(subset=['img_id'])
                
                # Extract defect columns (assuming anything after Images Name are defect columns until end)
                col_list = list(df.columns)
                img_col_idx = col_list.index(img_col)
                defect_cols = col_list[img_col_idx + 1:] # Get all remaining columns
                
                # Clean up defect columns
                valid_defect_cols = []
                for dc in defect_cols:
                    if 'Unnamed' not in str(dc) and str(dc).strip() != '' and str(dc) != 'img_id':
                        valid_defect_cols.append(dc)
                
                # Create a standardized dataframe
                std_df = pd.DataFrame()
                std_df['img_id'] = df['img_id']
                std_df['source_excel'] = os.path.basename(f)
                
                # Normalize defect labels to 0 or 1
                for dc in valid_defect_cols:
                    std_df[str(dc).strip()] = pd.to_numeric(df[dc], errors='coerce').fillna(0).astype(int)
                    
                all_dfs.append(std_df)
        except Exception as e:
            print(f"Warning parsing {os.path.basename(f)}: {e}")
            
    if not all_dfs:
        print("No labels extracted!")
        return
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Fill missing columns from concat with 0
    final_df = final_df.fillna(0)
    
    print(f"Total labeled Excel rows: {len(final_df)}")
    
    # Check matching
    final_df['image_path'] = final_df['img_id'].map(img_id_to_path)
    
    # Drop rows where we couldn't find the image
    matched_df = final_df.dropna(subset=['image_path']).copy()
    
    print(f"Successfully matched labels to images: {len(matched_df)}")
    
    # Create an 'is_damaged' column (1 if any defect column is > 0, else 0)
    defect_cols = [c for c in matched_df.columns if c not in ['img_id', 'source_excel', 'image_path']]
    
    # Calculate total defects per row
    matched_df['is_damaged'] = (matched_df[defect_cols].sum(axis=1) > 0).astype(int)
    
    print(f"Damaged items: {matched_df['is_damaged'].sum()}, Clean items: {len(matched_df) - matched_df['is_damaged'].sum()}")
    
    # Save the consolidated CSV
    csv_path = os.path.join(OUTPUT_DIR, 'labels.csv')
    matched_df.to_csv(csv_path, index=False)
    print(f"Saved consolidated labels to {csv_path}")
    
    print("\nDefect Categories Found:")
    for c in defect_cols:
        print(f" - {c}: {matched_df[c].sum()} samples")

if __name__ == "__main__":
    main()
