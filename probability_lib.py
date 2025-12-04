import numpy as np
import os
import csv
import paint_pmf
from scipy import stats
from scipy.stats import truncnorm


#=====================================================================================================================
def entropy(frequencies):
    """
    Calculate the entropy (uncertainty) of a probability mass function (PMF)
    
    Entropy formula: H(X) = -Σ p(x) * log2(p(x))
    where p(x) are the probabilities (frequencies)
    
    Input: frequencies - numpy array or list of probabilities (should sum to 1)
    Output: entropy value (in bits)
    
    Note: Handles p(x) = 0 by treating 0 * log2(0) = 0
    """
    
    # Convert to numpy array if needed
    frequencies = np.array(frequencies)
    
    # Filter out zero probabilities (0 * log2(0) = 0 by convention)
    non_zero_probs = frequencies[frequencies > 0]
    
    # Calculate entropy: -Σ p(x) * log2(p(x))
    entropy_value = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    return entropy_value


#=====================================================================================================================
def entropy_csv(csv_path, paintings, artist):
    """
    Calculate the entropy for every painting and write results to a CSV file.
    
    For each painting, gets its PMF and calculates the entropy of the color frequency
    distribution using the entropy() helper function. Writes results to a CSV file with
    columns: ENTROPY, PAINTING_NAME, FILENAME.
    
    Input:
        csv_path - path to the PMF CSV file
        paintings - list of tuples (filename, painting_name)
        artist - artist name (used for output CSV filename)
    """
    
    # Create CSV file for entropy results
    entropy_csv_path = "csv/" + artist + "_entropy.csv"
    
    # Delete CSV file if it exists to ensure fresh data
    if os.path.exists(entropy_csv_path):
        os.remove(entropy_csv_path)
    
    # Open CSV file for writing
    with open(entropy_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['ENTROPY', 'PAINTING_NAME', 'FILENAME'])
        
        # Define counts -- status print statements
        count = 0 
        completed_percent = 0.25 

        # Calculate entropy for each painting
        for filename, painting_name in paintings:

            # Calculate progress percentage -- status print statements
            count += 1
            if count / len(paintings) >= completed_percent:
                print(f"{round(completed_percent * 100)}% percent complete for {entropy_csv_path}. {count} entropy values computed.")
                completed_percent += 0.25 

            # Get PMF of a single painting
            single_sorted_colors, single_sorted_frequencies, single_painting_name, single_filename_path = paint_pmf.get_pmf(csv_path, painting_name)
            
            # Calculate entropy of the painting's color frequency distribution
            entropy_value = entropy(single_sorted_frequencies)
            
            # Write to CSV
            writer.writerow([entropy_value, single_painting_name, single_filename_path])
    
    print(f"\nEntropy values written to {entropy_csv_path}")


#=====================================================================================================================
def kl_divergence(painting_colors, painting_frequencies, overall_colors, overall_frequencies):
    """
    Calculate the Kullback-Leibler (KL) divergence between a single painting PMF and the overall artist PMF
    
    KL divergence formula: D_KL(P || Q) = Σ P(x) * log2(P(x) / Q(x))
    where P is the painting PMF and Q is the overall artist PMF
    
    This measures how different a painting's color distribution is from the artist's typical colors.
    Higher KL divergence = more different from the artist's usual style.
    
    Input:
        painting_colors - list of color tuples (r, g, b) for the single painting
        painting_frequencies - numpy array of probabilities for the painting (should sum to 1)
        overall_colors - list of color tuples (r, g, b) for the overall artist PMF
        overall_frequencies - numpy array of probabilities for the overall PMF (should sum to 1)
    Output: KL divergence value (in bits)
    
    Note: Handles cases where Q(x) = 0 by skipping those terms (infinite divergence would occur)
    """
    
    # Convert to numpy arrays
    painting_frequencies = np.array(painting_frequencies)
    overall_frequencies = np.array(overall_frequencies)
    
    # Create a dictionary mapping colors to probabilities for the overall PMF
    overall_pmf_dict = {}
    for color, freq in zip(overall_colors, overall_frequencies):
        overall_pmf_dict[color] = freq
    
    # Calculate KL divergence: D_KL(P || Q) = Σ P(x) * log2(P(x) / Q(x))
    kl_sum = 0.0
    
    for color, p_x in zip(painting_colors, painting_frequencies):
        # Skip if P(x) = 0 (contributes 0 to the sum)
        if p_x == 0:
            continue
        
        # Get Q(x) from the overall PMF
        if color in overall_pmf_dict:
            q_x = overall_pmf_dict[color]
            
            # Skip if Q(x) = 0 (would cause division by zero / infinite divergence)
            if q_x == 0:
                continue
            
            # Add contribution: P(x) * log2(P(x) / Q(x))
            kl_sum += p_x * np.log2(p_x / q_x)
        else:
            # Color exists in painting but not in overall PMF
            # This means Q(x) = 0, which would cause infinite divergence
            # We skip this term (or could return infinity, but skipping is more practical)
            continue
    
    return kl_sum


#=====================================================================================================================
def kl_divergence_csv(csv_path, paintings, artist, artist_sorted_colors, artist_sorted_frequencies):

    # Create CSV file for KL divergence results
    kl_csv_path = "csv/" + artist + "_kl_divergence.csv"
    
    # Delete CSV file if it exists to ensure fresh data
    if os.path.exists(kl_csv_path):
        os.remove(kl_csv_path)
    
    # Open CSV file for writing
    with open(kl_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['KL_DIVERGENCE', 'PAINTING_NAME', 'FILENAME'])
        
        # Define counts -- status print statements
        count = 0 
        completed_percent = 0.25 

        # Calculate KL divergence for each painting
        for filename, painting_name in paintings:

            # Calculate progress percentage -- status print statements
            count += 1 
            if count / len(paintings) >= completed_percent:
                print(f"{round(completed_percent * 100)}% percent complete for {kl_csv_path}. {count} KL divergence values computed.")
                completed_percent += 0.25 

            # Get PMF of a single painting
            single_sorted_colors, single_sorted_frequencies, single_painting_name, single_filename_path = paint_pmf.get_pmf(csv_path, painting_name)
            
            # Calculate KL divergence between single painting and artist overall PMF
            kl_div = kl_divergence(
                single_sorted_colors, 
                single_sorted_frequencies, 
                artist_sorted_colors, 
                artist_sorted_frequencies
            )
            
            # Write to CSV
            writer.writerow([kl_div, single_painting_name, single_filename_path])
    
    print(f"\nKL divergence values written to {kl_csv_path}")


#=====================================================================================================================
def get_max_min_kl(kl_csv_path):
    """
    Find the paintings with the highest and lowest KL divergence values from the CSV file
    
    If multiple paintings have the same max or min KL divergence value, only the first
    one encountered (first row in CSV) will be selected.
    
    Input: kl_csv_path - path to the KL divergence CSV file
    Output: tuple of (max_kl, max_kl_painting_name, max_kl_filename, 
                      min_kl, min_kl_painting_name, min_kl_filename)
    """
    
    # Load KL divergence values
    kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)
    
    # Find indices of highest and lowest KL divergence
    # np.argmax() and np.argmin() return the first index when there are ties
    max_kl_index = np.argmax(kl_divergence_values)
    min_kl_index = np.argmin(kl_divergence_values)
    
    # Read CSV to get painting names and filenames
    with open(kl_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # Extract painting name and filename for highest KL divergence
    # If multiple paintings have the same max value, this picks the first one
    max_kl_painting_name = rows[max_kl_index]['PAINTING_NAME']
    max_kl_filename = rows[max_kl_index]['FILENAME']
    max_kl = kl_divergence_values[max_kl_index]
    
    # Extract painting name and filename for lowest KL divergence
    # If multiple paintings have the same min value, this picks the first one
    min_kl_painting_name = rows[min_kl_index]['PAINTING_NAME']
    min_kl_filename = rows[min_kl_index]['FILENAME']
    min_kl = kl_divergence_values[min_kl_index]
    
    return max_kl, max_kl_painting_name, max_kl_filename, min_kl, min_kl_painting_name, min_kl_filename


#=====================================================================================================================
def get_max_min_entropy(entropy_csv_path):
    """
    Find the paintings with the highest and lowest entropy values from the CSV file
    
    If multiple paintings have the same max or min entropy value, only the first
    one encountered (first row in CSV) will be selected.
    
    Input: entropy_csv_path - path to the entropy CSV file
    Output: tuple of (max_entropy, max_entropy_painting_name, max_entropy_filename, 
                      min_entropy, min_entropy_painting_name, min_entropy_filename)
    """
    
    # Load entropy values
    entropy_values = np.loadtxt(entropy_csv_path, delimiter=',', skiprows=1, usecols=0)
    
    # Find indices of highest and lowest entropy
    # np.argmax() and np.argmin() return the first index when there are ties
    max_entropy_index = np.argmax(entropy_values)
    min_entropy_index = np.argmin(entropy_values)
    
    # Read CSV to get painting names and filenames
    with open(entropy_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # Extract painting name and filename for highest entropy
    # If multiple paintings have the same max value, this picks the first one
    max_entropy_painting_name = rows[max_entropy_index]['PAINTING_NAME']
    max_entropy_filename = rows[max_entropy_index]['FILENAME']
    max_entropy = entropy_values[max_entropy_index]
    
    # Extract painting name and filename for lowest entropy
    # If multiple paintings have the same min value, this picks the first one
    min_entropy_painting_name = rows[min_entropy_index]['PAINTING_NAME']
    min_entropy_filename = rows[min_entropy_index]['FILENAME']
    min_entropy = entropy_values[min_entropy_index]
    
    return max_entropy, max_entropy_painting_name, max_entropy_filename, min_entropy, min_entropy_painting_name, min_entropy_filename



#=====================================================================================================================
# EXPERIMENTAL FUNCTIONS FOR KL DISTRIBUTION
# If an artist created a new painting, what’s the probability they would try a different color palette compared to 
# their usual portfolio?
#=====================================================================================================================

#=====================================================================================================================
# def calculate_kl_norm(kl_csv_path):
#     """
#     Calculate a truncated normal distribution approximation from KL divergence values.
    
#     Reads all KL divergence values from the CSV file, calculates the mean and variance,
#     and returns a scipy.stats truncated normal distribution object (truncated at 0).
#     This ensures the PDF integrates to 1 over the valid range [0, +∞) since KL divergence
#     is always non-negative.
    
#     Input: kl_csv_path - path to the KL divergence CSV file
#     Output: scipy.stats truncnorm distribution object with calculated mean and std, truncated at 0
#     """
    
#     # Load KL divergence values from CSV (first column)
#     kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)
    
#     # Calculate mean and standard deviation
#     mean = np.mean(kl_divergence_values)
#     std = np.std(kl_divergence_values, ddof=1)  # ddof=1 for sample standard deviation
    
#     # Create truncated normal distribution (truncated at 0, since KL divergence ≥ 0)
#     # truncnorm parameters: a, b are the truncation points in standard deviations from the mean
#     # For truncation at 0: a = (0 - mean) / std, b = +infinity
#     a = (0 - mean) / std  # lower bound in standard deviations
#     b = np.inf  # upper bound (no upper truncation)
    
#     # Create truncated normal distribution
#     # loc=mean, scale=std, with truncation bounds a and b
#     truncated_normal_dist = truncnorm(a=a, b=b, loc=mean, scale=std)
    
#     return truncated_normal_dist


#=====================================================================================================================
# def pr_kl_norm(normal_dist, kl):

#     print(normal_dist.cdf(kl))


#=====================================================================================================================
# def calculate_kl_gamma(kl_csv_path):
#     """
#     Calculate a gamma distribution approximation from KL divergence values.
    
#     Reads all KL divergence values from the CSV file, fits a gamma distribution
#     to the data using maximum likelihood estimation, and returns a scipy.stats
#     gamma distribution object. The gamma distribution is a natural fit for KL
#     divergence values since it is defined for non-negative values (KL divergence ≥ 0)
#     and can model various shapes of positive continuous data.
    
#     Input: kl_csv_path - path to the KL divergence CSV file
#     Output: scipy.stats gamma distribution object with fitted parameters
#     """
    
#     # Load KL divergence values from CSV (first column)
#     kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)
    
#     # Fit gamma distribution to the data using maximum likelihood estimation
#     # gamma.fit() returns (shape, loc, scale) parameters
#     # We use floc=0 to fix the location parameter at 0 (since KL divergence ≥ 0)
#     shape, loc, scale = stats.gamma.fit(kl_divergence_values, floc=0)
    
#     # Create gamma distribution with fitted parameters
#     # shape parameter (a) and scale parameter
#     gamma_dist = stats.gamma(a=shape, loc=loc, scale=scale)
    
#     return gamma_dist
