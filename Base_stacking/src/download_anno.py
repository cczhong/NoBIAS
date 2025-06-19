import os
import requests
import zipfile
from tqdm import tqdm

# Note: The cif_files URL was empty, so it's not included in the download logic.
cif_files = ""

# Dictionary mapping annotation names to their download URLs
annotations_to_download = {
    "cif": "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/cif.zip?download=true",
    "CLARNA": "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/CLARNA.zip?download=true",
    "DSSR": "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/DSSR.zip?download=true",
    "FR3D": "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/FR3D.zip?download=true",
    "MCA": "https://huggingface.co/datasets/VatsaDev/BioD2/resolve/main/MCA.zip?download=true"
}

def download_all():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for name, url in annotations_to_download.items():
        output_dir = os.path.join(base_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        zip_path = os.path.join(output_dir, f"{name}.zip")
        try:

            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1kb blocks

            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {name}.zip") as pbar:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk: # filter out keep-alive new chunks
                            pbar.update(len(chunk))
                            f.write(chunk)

            print(f"Successfully downloaded to: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            os.remove(zip_path) # clean up zips

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {name}: {e}")
        except zipfile.BadZipFile:
            print(f"Error: The downloaded file for {name} is not a valid zip file.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {name}: {e}")

if __name__ == "__main__":
    download_all()