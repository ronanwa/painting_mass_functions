import data_loader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import paint_pmf
import numpy as np


#================================================================================
# SINGLE PAINTING PMF
#================================================================================
def display_painting(csv_path, single_painting_name, artist):
    # Plot a specific painting
    single_sorted_colors, single_sorted_frequencies, single_painting_name, single_filename_path = paint_pmf.get_pmf(csv_path, single_painting_name)
    
    plot_figure(
        len(single_sorted_colors), 
        single_sorted_frequencies, 
        single_sorted_colors, 
        1, 
        artist, 
        painting_name=single_painting_name, 
        filename_path=single_filename_path
        )

#=====================================================================================================================
def plot_figure(num_colors, sorted_frequencies, sorted_colors, n_paintings, artist_name, painting_name=0, filename_path=0):

    # Create figure with three subplots: image on left, histogram on right, color bar at bottom
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Image on top left
    ax2 = fig.add_subplot(gs[0, 1])  # Histogram on top right
    ax3 = fig.add_subplot(gs[1, :])  # Color bar spanning bottom
    
    # Display the image on the left if there is one
    if painting_name != 0 and filename_path != 0:
        img, img_arr = data_loader.load_img(filename_path)
        ax1.imshow(img)
        ax1.set_title(f'"{painting_name.replace('_', ' ').title()}"')
        
    ax1.axis('off')
    
    # Convert color strings to tuples if needed (for colors from add_pmf())
    rgb_colors = []
    for color in sorted_colors:
        if isinstance(color, str):
            # Parse string like "(0.5, 0.3, 0.8)" to tuple
            color_str = color.strip('()')
            rgb = tuple(float(x) for x in color_str.split(', '))
            rgb_colors.append(rgb)
        else:
            # Already a tuple
            rgb_colors.append(color)
    
    # Use num_colors or length of frequencies if num_colors not provided
    if num_colors is None or num_colors == 0:
        num_colors = len(sorted_frequencies)
    
    # Create bar plot with sorted data on the right
    bars = ax2.bar(range(num_colors), sorted_frequencies, edgecolor='black', linewidth=0.5)
    
    # Color each bar with its corresponding RGB color
    for i, bar in enumerate(bars):
        bar.set_facecolor(rgb_colors[i])

    artist_name = artist_name.replace('_', ' ').title()
    
    ax2.set_xlabel('Color (sorted by probability)')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Color PMF for {artist_name} ({num_colors} Colors across {int(n_paintings)} Painting(s))')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot color bar at the bottom
    # Filter out colors with frequency <= 0 and sort
    frequencies = np.array(sorted_frequencies)
    mask = frequencies > 0
    colors_filtered = [rgb_colors[i] for i in range(len(rgb_colors)) if mask[i]]
    frequencies_filtered = frequencies[mask]
    
    # Sort by frequency (largest to smallest)
    sorted_indices = np.argsort(frequencies_filtered)[::-1]
    colors_filtered = [colors_filtered[i] for i in sorted_indices]
    frequencies_filtered = frequencies_filtered[sorted_indices]
    
    # Draw color bar
    x_start = 0.0
    for color, freq in zip(colors_filtered, frequencies_filtered):
        rect = mpatches.Rectangle((x_start, 0), freq, 1, 
                                  facecolor=color, edgecolor='none', linewidth=0)
        ax3.add_patch(rect)
        x_start += freq
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Probability')
    ax3.set_ylabel('')
    ax3.set_title('PMF Color Bar (sorted by probability, largest to smallest)')
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()



#=====================================================================================================================
def plot_pmf_color_bar(sorted_colors, sorted_frequencies):
    """
    Plot a single horizontal bar representing a PMF as a 1D color bar.
    
    The bar is filled with colors whose frequency > 0, sorted from largest to smallest.
    The x-axis represents probability from 0 to 1.
    
    Input:
        sorted_colors - list of RGB tuples (r, g, b) in [0, 1] range
        sorted_frequencies - numpy array of frequencies/probabilities
    """
    
    # Convert to numpy array for easier manipulation
    frequencies = np.array(sorted_frequencies)
    
    # Filter out colors with frequency <= 0
    mask = frequencies > 0
    colors = [sorted_colors[i] for i in range(len(sorted_colors)) if mask[i]]
    frequencies = frequencies[mask]
    
    # Sort by frequency (largest to smallest) to ensure correct order
    sorted_indices = np.argsort(frequencies)[::-1]
    colors = [colors[i] for i in sorted_indices]
    frequencies = frequencies[sorted_indices]
    
    # Create figure with a single horizontal bar
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Starting x position
    x_start = 0.0
    
    # Draw each color segment as a rectangle
    for color, freq in zip(colors, frequencies):
        # Create a rectangle for this color segment
        rect = mpatches.Rectangle((x_start, 0), freq, 1, 
                                  facecolor=color, edgecolor='none', linewidth=0)
        ax.add_patch(rect)
        
        # Update starting position for next segment
        x_start += freq
    
    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_ylabel('')
    ax.set_title('PMF Color Bar (sorted by probability, largest to smallest)')
    
    # Remove y-axis ticks and labels since it's just a single bar
    ax.set_yticks([])
    
    # Add grid for better visualization
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

#=====================================================================================================================
def plot_kl_points(kl_csv_path):
    """
    Plot KL divergence values for all paintings as points.

    Reads the KL divergence CSV (first column KL_DIVERGENCE) and plots
    the values versus painting index.
    """

    # Load KL divergence values from CSV (first column)
    kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)

    # Sort values from smallest to largest, keep associated indices
    sorted_indices = np.argsort(kl_divergence_values)
    sorted_kl = kl_divergence_values[sorted_indices]

    # X-axis: rank index after sorting (0 = smallest KL)
    indices = np.arange(len(sorted_kl))

    # Create scatter/point plot
    plt.figure(figsize=(10, 4))
    plt.plot(indices, sorted_kl, 'o', markersize=3)
    plt.xlabel('Painting (sorted by KL, smallest → largest)')
    plt.ylabel('KL divergence (bits)')
    plt.title('KL divergence per painting')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#=====================================================================================================================
def plot_kl_norm(normal_dist, kl_csv_path=None):
    """
    Plot the truncated normal distribution approximation of KL divergence values.
    
    Creates two separate plots:
    1. Histogram of actual KL divergence values (if kl_csv_path is provided)
    2. Truncated normal distribution curve (truncated at 0)
    
    Input:
        normal_dist - scipy.stats truncnorm distribution object (truncated at 0)
        kl_csv_path - optional path to KL divergence CSV file for histogram plot
    """
    
    # Plot 1: Histogram of actual KL divergence values (if CSV path provided)
    if kl_csv_path is not None:
        kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(kl_divergence_values, bins=30, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        ax1.set_xlabel('KL Divergence (bits)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Histogram of Actual KL Divergence Values')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Plot 2: Normal distribution curve
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Determine x-axis range
    if kl_csv_path is not None:
        # Use range based on actual data
        x_min = np.min(kl_divergence_values)
        x_max = np.max(kl_divergence_values)
        x_range = x_max - x_min
        x_plot_min = max(0, x_min - 0.1 * x_range)  # Ensure non-negative
        x_plot_max = x_max + 0.1 * x_range
    else:
        # Use a reasonable range around the mean, but ensure non-negative
        mean = normal_dist.mean()
        std = normal_dist.std()
        x_plot_min = max(0, mean - 3 * std)  # Ensure non-negative
        x_plot_max = mean + 3 * std
    
    # Create x values for plotting the normal distribution
    x = np.linspace(x_plot_min, x_plot_max, 1000)
    
    # Calculate PDF values
    pdf_values = normal_dist.pdf(x)
    
    # Plot the truncated normal distribution
    ax2.plot(x, pdf_values, 'r-', linewidth=2, 
             label=f'Truncated Normal Distribution (μ={normal_dist.mean():.4f}, σ={normal_dist.std():.4f})')
    
    ax2.set_xlabel('KL Divergence (bits)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Truncated Normal Distribution Approximation of KL Divergence (truncated at 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_kl_gamma(kl_gamma, kl_csv_path=None):
    """
    Plot the gamma distribution approximation of KL divergence values.
    
    Creates two separate plots:
    1. Histogram of actual KL divergence values (if kl_csv_path is provided)
    2. Gamma distribution curve
    
    Input:
        kl_gamma - scipy.stats gamma distribution object
        kl_csv_path - optional path to KL divergence CSV file for histogram plot
    """
    
    # Plot 1: Histogram of actual KL divergence values (if CSV path provided)
    if kl_csv_path is not None:
        kl_divergence_values = np.loadtxt(kl_csv_path, delimiter=',', skiprows=1, usecols=0)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(kl_divergence_values, bins=30, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        ax1.set_xlabel('KL Divergence (bits)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Histogram of Actual KL Divergence Values')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Plot 2: Gamma distribution curve
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Determine x-axis range
    if kl_csv_path is not None:
        # Use range based on actual data
        x_min = np.min(kl_divergence_values)
        x_max = np.max(kl_divergence_values)
        x_range = x_max - x_min
        x_plot_min = max(0, x_min - 0.1 * x_range)  # Ensure non-negative
        x_plot_max = x_max + 0.1 * x_range
    else:
        # Use a reasonable range around the mean, but ensure non-negative
        mean = kl_gamma.mean()
        std = kl_gamma.std()
        x_plot_min = max(0, mean - 3 * std)  # Ensure non-negative
        x_plot_max = mean + 3 * std
    
    # Create x values for plotting the gamma distribution
    x = np.linspace(x_plot_min, x_plot_max, 1000)
    
    # Calculate PDF values
    pdf_values = kl_gamma.pdf(x)
    
    # Get distribution parameters for the label
    # For scipy.stats frozen distributions, parameters are stored in args and kwds
    try:
        # Try to get shape (a) from kwds or args
        if hasattr(kl_gamma, 'kwds') and 'a' in kl_gamma.kwds:
            shape = kl_gamma.kwds['a']
        elif hasattr(kl_gamma, 'args') and len(kl_gamma.args) > 0:
            shape = kl_gamma.args[0]
        else:
            raise AttributeError("Cannot find shape parameter")
        
        # Try to get scale from kwds
        if hasattr(kl_gamma, 'kwds') and 'scale' in kl_gamma.kwds:
            scale = kl_gamma.kwds['scale']
        else:
            scale = 1.0  # default scale
        
        label_text = f'Gamma Distribution (shape={shape:.4f}, scale={scale:.4f})'
    except (AttributeError, KeyError, IndexError):
        # Fallback to mean and std if parameter access fails
        mean = kl_gamma.mean()
        std = kl_gamma.std()
        label_text = f'Gamma Distribution (μ={mean:.4f}, σ={std:.4f})'
    
    # Plot the gamma distribution
    ax2.plot(x, pdf_values, 'r-', linewidth=2, label=label_text)
    
    ax2.set_xlabel('KL Divergence (bits)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Gamma Distribution Approximation of KL Divergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
