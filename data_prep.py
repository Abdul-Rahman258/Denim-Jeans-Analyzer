import pandas as pd
import glob
import os

excel_files = glob.glob(r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\Data\*.xlsx')
image_files = glob.glob(r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\Data\*.jpeg')
image_names = [os.path.basename(f) for f in image_files]

print(f"Total image files found: {len(image_files)}\n")

all_dfs = []
for f in excel_files:
    try:
        # Read without header
        df = pd.read_excel(f, header=None)
        
        # Find the row that contains "Images Name" (case-insensitive search)
        header_row_idx = None
        for idx, row in df.iterrows():
            if any(isinstance(val, str) and 'Images Name' in val for val in row.values):
                header_row_idx = idx
                break
                
        if header_row_idx is not None:
            # Re-read with actual header
            df = pd.read_excel(f, header=header_row_idx)
            # Find the actual images name column (might have spaces)
            img_col = [c for c in df.columns if 'Images Name' in str(c)][0]
            
            df = df.dropna(subset=[img_col])
            # Filter rows that look like actual image names
            df = df[df[img_col].astype(str).str.contains('IMG', na=False)]
            
            # Format image col to standard
            df['clean_img_name'] = df[img_col].astype(str).str.strip()
            df['source_file'] = os.path.basename(f)
            all_dfs.append(df)
            
    except Exception as e:
        print(f"Error on {f}: {e}")

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total labeled rows in Excel: {len(combined_df)}")
    
    excel_images = combined_df['clean_img_name'].tolist()
    
    matches = sum(1 for img in excel_images if img in image_names)
    print(f"Exact Matches: {matches}")
    
    # Try cleaning dir images to see if they match excel format
    # Excel format seems to be 'IMG_ (1).jpeg' but dir has 'IMG_1.jpg(1).jpeg' or 'IMG_10.jpg.jpeg'
    # Wait, maybe Excel has exactly what the dir has?
    print("\nSample Excel Image Names:")
    print(excel_images[:10])
    
    print("\nSample Directory Image Names:")
    print(image_names[:10])
else:
    print("Could not parse excel files.")
