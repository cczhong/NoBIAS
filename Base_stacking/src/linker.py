import pandas as pd
import os
import json
import logging
from tqdm.auto import tqdm

DATA_DIR = r""

# List of tuples: (relative path to CSV file, relative path to image folder)
# Paths are relative to DATA_DIR
SUBFOLDER_PAIRS = [
    (
        "pass_all.csv",
        r"output_images/pass_all" # Adjusted path relative to DATA_DIR
    ),
]

OUTPUT_JSONL_FOLDER = r"jsonl_data_full"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def construct_filename(row):
    """Constructs the expected image filename based on a CSV row."""
    try:
        # Ensure data types are correct before formatting
        mol_name = str(row['mol_name']).strip()
        chain1 = str(row['chain1']).strip()
        # Read as float first for safety against decimals, then convert to int
        position1 = int(float(row['position1']))
        chain2 = str(row['chain2']).strip()
        position2 = int(float(row['position2']))

        # Format according to the image generation convention
        filename = f"{mol_name}_{chain1}{position1}_{chain2}{position2}.png"
        return filename
    except (KeyError, ValueError, TypeError) as e:
        # Log the specific row content causing the error for easier debugging
        logging.error(f"Error constructing filename. Row data: {row.to_dict()}. Error: {e}")
        return None

def create_simple_hf_dataset_jsonl(base_data_dir, subfolder_pairs, output_jsonl_base_folder):
    """
    Generates separate JSONL files for each CSV/image pair.
    Processes each row independently. If an image exists, its relative path and
    label are written to the JSONL. If an image doesn't exist, the row is skipped.

    Args:
        base_data_dir (str): The absolute path to the base directory containing data.
        subfolder_pairs (list): A list of tuples, where each tuple contains
                                (relative_path_to_csv, relative_path_to_image_folder).
        output_jsonl_base_folder (str): The path to the directory where the output
                                         JSONL files will be saved.
    """
    overall_processed_count = 0
    overall_skipped_missing_img = 0
    overall_skipped_processing_error = 0
    overall_rows_evaluated = 0 # Total rows checked in all CSVs
    created_files = []

    # --- Validate base paths ---
    if not os.path.isdir(base_data_dir):
        logging.error(f"Base data directory not found or is not a directory: {base_data_dir}")
        print(f"Error: Base data directory not found: {base_data_dir}")
        return

    # --- Ensure the output JSONL directory exists ---
    try:
        os.makedirs(output_jsonl_base_folder, exist_ok=True)
        if not os.path.isdir(output_jsonl_base_folder):
             logging.error(f"Output path exists but is not a directory: {output_jsonl_base_folder}")
             print(f"Error: Output path exists but is not a directory: {output_jsonl_base_folder}")
             return
    except OSError as e:
        logging.error(f"Failed to create output directory {output_jsonl_base_folder}: {e}")
        print(f"Error: Could not create output directory: {output_jsonl_base_folder}")
        return


    # --- Process each pair ---
    for csv_relative_path, image_relative_folder in subfolder_pairs:
        # Normalize relative paths (handles mixed separators like / and \)
        csv_relative_path = os.path.normpath(csv_relative_path)
        image_relative_folder = os.path.normpath(image_relative_folder)

        # Construct absolute paths
        absolute_csv_path = os.path.join(base_data_dir, csv_relative_path)
        absolute_image_folder = os.path.join(base_data_dir, image_relative_folder)

        logging.info(f"\n--- Processing Pair ---")
        logging.info(f"CSV:          {absolute_csv_path}")
        logging.info(f"Image Folder: {absolute_image_folder}")

        # Determine output JSONL filename
        csv_basename = os.path.basename(absolute_csv_path)
        jsonl_filename = os.path.splitext(csv_basename)[0] + '.jsonl'
        output_jsonl_path = os.path.join(output_jsonl_base_folder, jsonl_filename)
        logging.info(f"Output JSONL: {output_jsonl_path}")

        # Reset counters for this specific file
        processed_count = 0
        skipped_missing_img = 0
        skipped_processing_error = 0
        rows_evaluated = 0

        # --- Check input files ---
        if not os.path.isfile(absolute_csv_path):
            logging.error(f"CSV file not found or is not a file: {absolute_csv_path}. Skipping this pair.")
            continue
        if not os.path.isdir(absolute_image_folder):
            logging.error(f"Image folder not found or is not a directory: {absolute_image_folder}. Skipping this pair.")
            continue

        # --- Read CSV ---
        try:
            # Use dtype=str for flexibility, handle conversions later
            df = pd.read_csv(absolute_csv_path, dtype=str)
            logging.info(f"Loaded {len(df)} rows from {csv_basename}")
        except Exception as e:
            logging.error(f"Failed to read CSV {absolute_csv_path}: {e}. Skipping this pair.")
            continue

        # --- Check for required columns ---
        required_cols = ['mol_name', 'chain1', 'position1', 'chain2', 'position2']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logging.error(f"CSV {csv_basename} missing required columns: {missing_cols}. Skipping this pair.")
            continue
        if 'label' not in df.columns:
             logging.warning(f"CSV {csv_basename} missing optional 'label' column. Will use 'UNKNOWN' as placeholder.")
             # No need to add the column to the DataFrame, just handle it during processing

        # --- Process rows and write to JSONL ---
        try:
            with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                logging.info(f"Iterating through rows and writing to {jsonl_filename}...")
                # Use tqdm for progress bar
                pbar = tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_basename}", leave=False)
                for index, row in pbar:
                    rows_evaluated += 1
                    expected_filename = construct_filename(row)

                    if expected_filename is None:
                        skipped_processing_error += 1
                        # Error already logged in construct_filename
                        continue # Move to the next row

                    # Construct absolute path to check if image exists
                    absolute_image_path = os.path.join(absolute_image_folder, expected_filename)

                    # Check if the image file exists
                    if os.path.exists(absolute_image_path):
                        # Image exists! Process it.
                        try:
                            # Extract the class label, handle potential missing 'label' column
                            class_label = str(row.get('label', 'UNKNOWN')).strip() # Use get with default

                            # Store RELATIVE path in the JSONL using forward slashes
                            # Combine the relative image folder path with the filename
                            relative_image_path = os.path.join(image_relative_folder, expected_filename).replace("\\", "/")

                            # Create the data record
                            data_record = {
                                "file_path": relative_image_path, # Relative to DATA_DIR
                                "class_category": class_label
                                # Add other metadata if needed:
                                # "mol_name": str(row.get('mol_name')),
                                # ...
                            }

                            # Write the JSON record to the file, followed by a newline
                            outfile.write(json.dumps(data_record) + '\n')
                            processed_count += 1

                        except Exception as e:
                             # Log error specific to processing this valid image row
                             logging.error(f"Error processing row {index} (Image found: {absolute_image_path}): {e} | Row data: {row.to_dict()}")
                             skipped_processing_error += 1
                    else:
                        # Image not found - Just log a warning and skip this row
                        logging.warning(f"Image file not found, skipping row {index}: {absolute_image_path}")
                        skipped_missing_img += 1

                # Close tqdm cleanly after the loop for this file finishes
                pbar.close()

            # Post-processing log for the current CSV
            logging.info(f"Finished processing {csv_basename}.")
            logging.info(f"  Total rows evaluated: {rows_evaluated}")
            logging.info(f"  Entries written (images found): {processed_count}")
            logging.info(f"  Rows skipped (missing images): {skipped_missing_img}")
            logging.info(f"  Rows skipped (processing errors): {skipped_processing_error}")

            if processed_count > 0 or not df.empty or skipped_missing_img > 0 or skipped_processing_error > 0: # Check if any work was attempted
                 created_files.append(output_jsonl_path)

            # Update overall totals
            overall_rows_evaluated += rows_evaluated
            overall_processed_count += processed_count
            overall_skipped_missing_img += skipped_missing_img
            overall_skipped_processing_error += skipped_processing_error

        except IOError as e:
            logging.error(f"Could not write to output file {output_jsonl_path}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {csv_basename}: {e}", exc_info=True) # Add traceback


    logging.info("\n--- Overall Summary ---")
    logging.info(f"Base Data Directory: {base_data_dir}")
    logging.info(f"Output JSONL Folder: {output_jsonl_base_folder}")
    logging.info(f"Total CSV rows evaluated across all files: {overall_rows_evaluated}")
    logging.info(f"Total entries written across all JSONL files (images found): {overall_processed_count}")
    logging.info(f"Total skipped rows due to missing image files: {overall_skipped_missing_img}")
    logging.info(f"Total skipped rows due to processing errors: {overall_skipped_processing_error}")
    logging.info(f"Created/Attempted JSONL files:")
    if created_files:
        for f in created_files:
            logging.info(f"  - {f}")
    else:
        logging.warning("No JSONL files were created or processed. Check logs for errors.")
    logging.info("-------------------------")
    return created_files # Return list of created files

# --- Run the Script ---
if __name__ == "__main__":
    # Run the main function with the configured paths
    created_files = create_simple_hf_dataset_jsonl(DATA_DIR, SUBFOLDER_PAIRS, OUTPUT_JSONL_FOLDER)

    # --- Print helpful loading instructions ---
    print(f"\nScript finished.")
    if created_files:
        print(f"Output JSONL files written to: {OUTPUT_JSONL_FOLDER}")
        # Use os.path.abspath to show the full path for clarity
        abs_data_dir = os.path.abspath(DATA_DIR)
        abs_jsonl_folder = os.path.abspath(OUTPUT_JSONL_FOLDER)
        print(f"\nThe 'file_path' column in the JSONL files contains paths relative to the base data directory:")
        print(f"  Base Data Dir: {abs_data_dir}")

        # Generate loading examples using the *absolute* paths determined above
        # Use repr() to get a string representation that includes necessary escaping (like raw strings)
        print(f"\nTo load one of these datasets using Hugging Face datasets:")
        print(f"from datasets import load_dataset, Image")
        print(f"import os")
        print(f"DATA_DIR = {repr(abs_data_dir)}")
        print(f"JSONL_FOLDER = {repr(abs_jsonl_folder)}")

        # Example using the first created file
        if created_files: # Ensure list is not empty
            first_file_abs = os.path.abspath(created_files[0])
            print(f"\n# Example: Load the first created dataset ({os.path.basename(first_file_abs)})")
            print(f"jsonl_file_to_load = {repr(first_file_abs)}")
            print(f"# The data_dir argument tells datasets where to find the relative paths from the JSONL")
            print(f"# Make sure DATA_DIR points to the correct base location: {repr(abs_data_dir)}")
            print(f"try:")
            print(f"    dataset = load_dataset('json', data_files=jsonl_file_to_load, field='file_path', data_dir=DATA_DIR)")
            print(f"    # Cast the column containing the relative path to Image objects")
            print(f"    dataset = dataset.cast_column('file_path', Image())")
            print(f"    # Example access:")
            print(f"    # print(dataset['train'][0]['file_path']) # This will now be an Image object")
            print(f"    # print(dataset['train'][0]['class_category'])")
            print(f"except Exception as e:")
            print(f"    print(f'Error loading dataset: {{e}}')")
            print(f"    print('Check that DATA_DIR is correct and images exist at the relative paths specified in the JSONL.')")


            print(f"\n# To load ALL jsonl files in the directory as one dataset:")
            # Use file globbing within the absolute path
            all_jsonl_pattern = os.path.join(abs_jsonl_folder, '*.jsonl')
            print(f"all_jsonl_files_pattern = {repr(all_jsonl_pattern)}")
            print(f"# Use the pattern directly in data_files")
            print(f"try:")
            print(f"    combined_dataset = load_dataset('json', data_files=all_jsonl_files_pattern, field='file_path', data_dir=DATA_DIR)")
            print(f"    combined_dataset = combined_dataset.cast_column('file_path', Image())")
            print(f"    # print(combined_dataset['train'])")
            print(f"except Exception as e:")
            print(f"    print(f'Error loading combined dataset: {{e}}')")
            print(f"    print('Check that DATA_DIR is correct and images exist at the relative paths specified in the JSONL files.')")

    else:
        print("No JSONL files were created. Please check the console output and logs for errors.")
