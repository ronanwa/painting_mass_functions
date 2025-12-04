from PIL import Image
import numpy as np
import csv
import os
import paint_pmf


#===================================================================================================
def create_filenames(data_folder='data'):
    """
    Scan the data folder and get all image files with their relative paths
    Input: data_folder (path to the folder containing paintings)
    Output: 2D array where each entry is [filepath, painting_name]
    """
    filename_array = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Only process .jpg files
            if file.lower().endswith('.jpg'):
                # Get relative path from project root
                relative_path = os.path.join(root, file)
                
                # Create painting name from filename (remove .jpg extension)
                painting_name = os.path.splitext(file)[0]
                
                # Add to array
                filename_array.append([relative_path, painting_name])
    
    return filename_array


#===================================================================================================
def create_csv(paintings, csv_path):

    # Delete CSV file if it exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Define counts -- status print statements
    count = 0  
    completed_percent = 0.25 

    for filename, painting_name in paintings:
        
        # Calculate progress percentage -- status print statements
        count += 1 
        if count / len(paintings) >= completed_percent:
            print(f"{round(completed_percent * 100)}% complete for {csv_path}. {count} paintings processed.")
            completed_percent += 0.25 
       
        # Load image
        img, img_array = load_img(filename)

        # Get color PMF from the painting
        num_colors, sorted_frequencies, sorted_colors = paint_pmf.create_pmf(img, img_array, painting_name, n_bins=6)

        # Add painting PMF to CSV file
        write_csv(sorted_frequencies, sorted_colors, painting_name, filename, csv_path)


#=====================================================================================================================
def write_csv(sorted_frequencies, sorted_colors, painting_name, filename, filepath):
    """
    Append to a CSV file with columns: painting name, R, G, B, probability, and filename
    Input: sorted_frequencies, sorted_colors, painting_name, filename, filepath
    Output: Appends to CSV file with PAINTING_NAME, R (float32), G (float32), B (float32), PROBABILITY, FILENAME columns
    """
    
    # Create csv directory if it doesn't exist
    os.makedirs('csv', exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(filepath)
    
    # Append to CSV file
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if file is new
        if not file_exists:
            writer.writerow(['R', 'G', 'B', 'PROBABILITY', 'PAINTING_NAME', 'FILENAME'])
        # Write data rows: R, G, B (as float32), frequency, painting name, filename
        for freq, color in zip(sorted_frequencies, sorted_colors):
            r = np.float32(color[0])
            g = np.float32(color[1])
            b = np.float32(color[2])
            writer.writerow([r, g, b, freq, painting_name, filename])
    
    return filepath


#=====================================================================================================================
def load_img(filename):
    """
    # Load image of painting
    # Input: File path of image
    # Output: image; numpy array of RGB channels
    """

    # Load image
    img = Image.open(filename)

    # Convert image to numpy array
    img_array = np.array(img)

    # Return image and image array
    return img, img_array


#=====================================================================================================================
def clean_filenames(data_folder='data'):
    """
    Walk through all files in the data folder and rename them:
    - Convert all characters to lowercase
    - Replace spaces with underscores
    Input: data_folder (path to the folder containing files)
    Output: None (renames files in place)
    """
    
    renamed_count = 0
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Get the original file path
            original_path = os.path.join(root, file)
            
            # Create new filename: lowercase and replace spaces with underscores
            new_filename = file.lower().replace(' ', '_')
            
            # Only rename if the filename changed
            if new_filename != file:
                new_path = os.path.join(root, new_filename)
                
                # Check if target file already exists
                if os.path.exists(new_path):
                    print(f"Warning: {new_path} already exists, skipping {original_path}")
                    continue
                
                # Rename the file
                os.rename(original_path, new_path)
                renamed_count += 1
                # print(f"Renamed: {file} -> {new_filename}")
    
    print(f"\nTotal files renamed: {renamed_count}")
    