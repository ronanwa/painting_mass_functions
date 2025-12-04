import numpy as np
import csv


#=====================================================================================================================
def create_pmf(img, img_array, painting_name,  n_bins=7):

    # Discretize RGB values to approximately 100 colors (5x5x5 = 125, closest with equal bins)
    # Quantize each channel: R=5 bins, G=5 bins, B=5 bins
    r_bins = n_bins
    g_bins = n_bins
    b_bins = n_bins
    
    # Quantize each channel using proper formula: value * bins // 256
    # Convert to int32 first to avoid uint8 overflow
    r_quantized = np.clip((img_array[:, :, 0].astype(np.int32) * r_bins) // 256, 0, r_bins - 1)
    g_quantized = np.clip((img_array[:, :, 1].astype(np.int32) * g_bins) // 256, 0, g_bins - 1)
    b_quantized = np.clip((img_array[:, :, 2].astype(np.int32) * b_bins) // 256, 0, b_bins - 1)
    
    # Combine into a single color index (0-26 for 3x3x3 = 27 colors)
    num_colors = r_bins * g_bins * b_bins
    color_indices = r_quantized * (g_bins * b_bins) + g_quantized * b_bins + b_quantized
    color_indices = color_indices.flatten()
    
    # Compute RGB colors for each color index
    bar_colors = []
    for idx in range(num_colors):
        # Reverse quantization to get RGB values
        b_quant = idx % b_bins
        g_quant = (idx // b_bins) % g_bins
        r_quant = idx // (g_bins * b_bins)
        
        # Convert back to RGB (use middle of each bin)
        # Each bin covers 256/bins values, so middle is at (quant + 0.5) * (256/bins)
        r_val = int((r_quant + 0.5) * (256 / r_bins))
        g_val = int((g_quant + 0.5) * (256 / g_bins))
        b_val = int((b_quant + 0.5) * (256 / b_bins))
        
        # Clamp to valid range
        r_val = min(255, max(0, r_val))
        g_val = min(255, max(0, g_val))
        b_val = min(255, max(0, b_val))
        
        # Normalize to [0, 1] for matplotlib
        bar_colors.append((r_val / 255.0, g_val / 255.0, b_val / 255.0))
    
    # Count frequencies of each color index
    unique_indices, counts = np.unique(color_indices, return_counts=True)
    
    # Normalize counts to get density
    total_pixels = len(color_indices)
    frequencies = counts / total_pixels
    
    # Create a full array for all possible color indices (some may have 0 frequency)
    full_frequencies = np.zeros(num_colors)
    for idx, freq in zip(unique_indices, frequencies):
        full_frequencies[idx] = freq
    
    # Sort by frequency (descending) - largest to smallest
    sorted_indices = np.argsort(full_frequencies)[::-1]
    sorted_frequencies = full_frequencies[sorted_indices]
    sorted_colors = [bar_colors[i] for i in sorted_indices]

    return num_colors, sorted_frequencies, sorted_colors
    

#=====================================================================================================================
def add_pmf(csv_filename):
    """
    Sum all frequencies with respect to each color across all paintings and normalize to create a PMF
    Input: csv_filename (path to CSV file with PMF data)
    Output: tuple of (sorted_colors, normalized_frequencies) - arrays sorted by frequency (descending), frequencies normalized to sum to 1
    """
    
    # Dictionary to store summed frequencies for each color
    color_freq_dict = {}
    
    # First, count total rows for progress tracking
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        total_rows = sum(1 for row in reader)
    
    # Read CSV file and process
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Define counts -- status print statements
        count = 0 
        completed_percent = 0.25 

        # For every row in csv file
        for row in reader:
            
            # Calculate progress percentage -- status print statements
            count += 1
            if count / total_rows >= completed_percent:
                print(f"{round(completed_percent * 100)}% percent complete for {csv_filename}. {count} color frequncies added.")
                completed_percent += 0.25  
            
            # Read R, G, B as separate columns and convert to float32
            r = np.float32(row['R'])
            g = np.float32(row['G'])
            b = np.float32(row['B'])
            frequency = float(row['PROBABILITY'])
            
            # Create color tuple as key for dictionary
            color = (r, g, b)
            
            # Sum frequencies for each color
            if color in color_freq_dict:
                color_freq_dict[color] += frequency
            else:
                color_freq_dict[color] = frequency
    
    # Convert to lists and sort by frequency (descending)
    colors = list(color_freq_dict.keys())
    frequencies = [color_freq_dict[color] for color in colors]
    
    # Sort by frequency (descending)
    sorted_indices = np.argsort(frequencies)[::-1]
    sorted_colors = [colors[i] for i in sorted_indices]
    summed_frequencies = np.array([frequencies[i] for i in sorted_indices])
    
    # Normalize frequencies to sum to 1 (create PMF)
    total_sum = np.sum(summed_frequencies)
    normalized_frequencies = summed_frequencies / total_sum
    
    # Calculate number of paintings in file
    n_paintings = count / len(normalized_frequencies)

    return sorted_colors, normalized_frequencies, n_paintings


#=====================================================================================================================
def get_pmf(csv_filename, painting_name):
    """
    Get all rows for a specific painting from the CSV file
    Input: csv_filename (path to CSV file), painting_name (name of the painting to retrieve)
    Output: tuple of (colors, frequencies, painting_name, filename_path)
        - colors: list of tuples (r, g, b) as float32
        - frequencies: numpy array of frequencies
        - painting_name: the painting name (from CSV)
        - filename_path: the file path (same for all rows of a painting)
    """
    
    colors = []
    frequencies = []
    filename_path = None
    
    # Read CSV file and filter for the specified painting
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Check if this row belongs to the specified painting
            if row['PAINTING_NAME'] == painting_name:
                # Read R, G, B as separate columns and convert to float32
                r = np.float32(row['R'])
                g = np.float32(row['G'])
                b = np.float32(row['B'])
                frequency = float(row['PROBABILITY'])
                
                # Store color as tuple
                color = (r, g, b)
                colors.append(color)
                frequencies.append(frequency)
                
                # Store filename path (same for all rows, so just get it once)
                if filename_path is None:
                    filename_path = row['FILENAME']
    
    # Convert frequencies to numpy array
    frequencies = np.array(frequencies)
    
    # Sort by frequency (descending) - largest to smallest
    sorted_indices = np.argsort(frequencies)[::-1]
    sorted_frequencies = frequencies[sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    return sorted_colors, sorted_frequencies, painting_name, filename_path