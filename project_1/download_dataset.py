import os
import zipfile
import urllib.request

def main():
    # Define directories
    raw_dir = 'raw'
    gold_dir = 'gold'
    outputs_dir = 'outputs'
    
    for d in [raw_dir, gold_dir, outputs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
            
    # URLs to try (both old and new UCI repository links)
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    ]
    
    zip_path = os.path.join(raw_dir, 'student.zip')
    success = False
    
    for url in urls:
        try:
            print(f"Attempting to download from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download successful!")
            success = True
            break
        except Exception as e:
            print(f"Failed download from {url}: {e}")
            
    if not success:
        print("Error: Could not download the dataset. Please check your internet connection.")
        return
        
    # Unzip the downloaded file
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract student-mat.csv
        zip_ref.extract('student-mat.csv', raw_dir)
        print("Extracted student-mat.csv to raw/")
        
        # Check if student-mat.csv has semicolon delimiter and rename if needed
        # We also want to keep the original copy untouched as raw/original_dataset.csv
        original_csv = os.path.join(raw_dir, 'original_dataset.csv')
        extracted_csv = os.path.join(raw_dir, 'student-mat.csv')
        if os.path.exists(extracted_csv):
            import shutil
            shutil.copyfile(extracted_csv, original_csv)
            print(f"Copied student-mat.csv to {original_csv}")
            
    # Clean up zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Removed student.zip")
        
    print("Dataset setup completed successfully!")

if __name__ == '__main__':
    main()
