import os
import requests
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from urllib.parse import urlparse

# --- List of files to download ---
# You can add or remove URLs here.
# Example URLs: a zip file and a single file.
FILE_URLS = [
    "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/cif.zip?download=true",
    "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/output_fr3.csv?download=true",
    "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/base_pairs.zip?download=true"
]

def download_file(url):
    """Downloads a single file from a URL and shows a progress bar."""
    local_filename = os.path.basename(urlparse(url).path)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f, tqdm(
                desc=f"Downloading {local_filename}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        
        print(f"Finished downloading {local_filename}")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def unzip_and_cleanup(filepath):
    """
    Unzips a file and handles the common "double-directory" problem.
    """
    if not filepath.lower().endswith('.zip'):
        return

    print(f"Extracting {filepath}...")
    # Extract to a temporary directory named after the zip file
    extract_dir = os.path.splitext(filepath)[0]
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        # Get a list of all top-level members in the zip
        top_level_members = {member.split('/')[0] for member in zip_ref.namelist()}
        
        # Check if there is only one top-level directory inside
        is_single_root_dir = len(top_level_members) == 1 and all(
            m.startswith(list(top_level_members)[0] + '/') for m in zip_ref.namelist() if '/' in m
        )

        if is_single_root_dir:
            temp_extract_dir = extract_dir + "_temp_extract"
            os.makedirs(temp_extract_dir, exist_ok=True)
            zip_ref.extractall(temp_extract_dir)
            
            # The single root directory inside the temp directory
            single_root = os.path.join(temp_extract_dir, list(top_level_members)[0])
            
            # Move contents of the single root directory to the final destination
            # and remove the now-empty root and temp directories.
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir) # Remove if it exists to avoid merging issues
            shutil.move(single_root, extract_dir)
            shutil.rmtree(temp_extract_dir)
            print(f"Unzipped and fixed single root folder structure for {extract_dir}")
        else:
            # Otherwise, just extract directly
            zip_ref.extractall(extract_dir)
            print(f"Unzipped directly to {extract_dir}")

    # Remove the original zip file after successful extraction
    try:
        os.remove(filepath)
        print(f"Removed original zip file: {filepath}")
    except OSError as e:
        print(f"Error removing zip file {filepath}: {e}")


if __name__ == "__main__":
    # Use a ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all download tasks
        future_to_url = {executor.submit(download_file, url): url for url in FILE_URLS}
        
        # Process results as they complete
        for future in tqdm(future_to_url, total=len(FILE_URLS), desc="Overall Progress"):
            try:
                downloaded_filepath = future.result()
                if downloaded_filepath:
                    unzip_and_cleanup(downloaded_filepath)
            except Exception as e:
                print(f"An error occurred for URL {future_to_url[future]}: {e}")

    print("\nAll tasks complete.")