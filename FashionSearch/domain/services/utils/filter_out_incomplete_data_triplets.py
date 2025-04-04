import os
import json

def filter_entries_by_images(
    cap_folder: str, 
    image_folder: str, 
    dress_types: list, 
    output_folder: str
):
    """
    Reads each cap_{dress_type}_val.json file in `cap_folder`, 
    checks if candidate and target images exist in `image_folder`,
    and writes a filtered JSON file to `output_folder`.
    
    Args:
        cap_folder (str): Path to the folder containing the JSON caption files.
        image_folder (str): Path to the folder containing .jpg images.
        dress_types (list): List of dress types, e.g. ["dress", "shirt", "toptee"].
        output_folder (str): Path to the folder where filtered JSON files will be saved.
    """
    # Ensure output_folder exists
    os.makedirs(output_folder, exist_ok=True)

    for dt in dress_types:
        input_filename = f"cap_{dt}_train.json"
        input_path = os.path.join(cap_folder, input_filename)
        
        # Read the JSON file
        with open(input_path, "r") as f:
            entries = json.load(f)

        filtered_entries = []
        for entry in entries:
            candidate = entry["candidate"]
            target = entry["target"]
            
            # Build image file paths
            candidate_image_path = os.path.join(image_folder, f"{candidate}.jpg")
            target_image_path = os.path.join(image_folder, f"{target}.jpg")
            
            # Check if both exist
            if os.path.exists(candidate_image_path) and os.path.exists(target_image_path):
                filtered_entries.append(entry)

        # Write filtered entries to a new JSON file
        output_filename = f"cap_{dt}_train.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, "w") as f:
            json.dump(filtered_entries, f, indent=2)

        print(f"Filtered {len(filtered_entries)} entries in {output_filename}.")

# Example usage
if __name__ == "__main__":
    cap_folder = "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/captions"      # e.g., "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/cap"
    image_folder = "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/images" # e.g., "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/images"
    dress_types = ["dress", "shirt", "toptee"]
    output_folder = "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/test" # e.g., "C:/Users/Admin/Downloads/Malshini/MSC/MSC/dataset/cap_filtered"

    filter_entries_by_images(cap_folder, image_folder, dress_types, output_folder)
