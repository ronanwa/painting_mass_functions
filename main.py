import paint_pmf
import data_loader
import probability_lib
import visualize


def main():

    #================================================================================
    # NAMING AND PATHS
    #================================================================================

    # Artist name
    artist = "van_gogh"

    # CSV path
    csv_path = "csv/" + artist + "_pmf.csv"

    # Entropy CSV path
    entropy_csv_path = "csv/" + artist + "_entropy.csv"

    # Load KL divergence values from CSV (column at index 0)
    kl_csv_path = "csv/" + artist + "_kl_divergence.csv"

    #================================================================================
    # DATA LOADING
    #================================================================================

    # Clean file names
    print("\nCleaning filenames...")
    data_loader.clean_filenames("data/" + artist)
    print("Filenames successfully cleaned.")

    # Create filenames array
    print("\nCreating filenames array...")
    paintings = data_loader.create_filenames("data/" + artist)
    print("Filenames array successfully created.")


    #================================================================================
    # CREATE PMF OF EVERY PAINTING, WRITE TO CSV
    #================================================================================

    # Create PMF of every painting, write to CSV file
    print("\nCreating PMF CSV...")
    data_loader.create_csv(paintings, csv_path)
    print("PMF CSV sucessfully created.")


    #================================================================================
    # SUM ALL PMFs INTO ONE
    #================================================================================
    # # Add PMFs together
    print("\nStarting PMF summation...")
    artist_sorted_colors, artist_sorted_frequencies, artist_n_paintings = paint_pmf.add_pmf(csv_filename=csv_path)
    print("PMFs successfully added together.")

    # Plot PMF for all color frequncies across all paintings
    visualize.plot_figure(len(artist_sorted_colors), artist_sorted_frequencies, artist_sorted_colors, artist_n_paintings, artist)


    #================================================================================
    # SINGLE PAINTING PMF
    #================================================================================
    # # Plot a specific painting

    # visualize.display_painting(csv_path, "woman_sitting_by_a_cradle", artist)
    # visualize.display_painting(csv_path, "starry_night", artist)
    # visualize.display_painting(csv_path, "blossoming_almond_tree", artist)
    # visualize.display_painting(csv_path, "the_bedroom", artist)
    
    
    #================================================================================
    # ENTROPY
    #================================================================================

    print("\nEntropy Calculation:")
    # Calculate entropy for painting pmf(s)
    entropy = probability_lib.entropy(artist_sorted_frequencies)
    print(f"Entropy for Artist PMF: {probability_lib.entropy(artist_sorted_frequencies)}")
    
    # single_sorted_colors, single_sorted_frequencies, single_painting_name, single_filename_path = paint_pmf.get_pmf(csv_path, "basket_of_potatoes_2")
    # print(f"Entropy for {single_painting_name}: {probability_lib.entropy(single_sorted_frequencies)}")

    # single_sorted_colors, single_sorted_frequencies, single_painting_name, single_filename_path = paint_pmf.get_pmf(csv_path, "blossoming_chestnut_branches")
    # print(f"Entropy for {single_painting_name}: {probability_lib.entropy(single_sorted_frequencies)}")
    

    print("\nWriting Entropy CSV...")
    probability_lib.entropy_csv(csv_path, paintings, artist)
    print("KL Divergence CSV successfully written.")

    # Get max and min entropy paintings
    max_entropy, max_entropy_painting_name, max_entropy_filename, min_entropy, min_entropy_painting_name, min_entropy_filename = probability_lib.get_max_min_entropy(entropy_csv_path)

    print(f"\nHighest Entropy: {max_entropy:.6f} bits")
    print(f"  Painting: {max_entropy_painting_name}")
    print(f"  Filename: {max_entropy_filename}")
    print(f"\nLowest Entropy: {min_entropy:.6f} bits")
    print(f"  Painting: {min_entropy_painting_name}")
    print(f"  Filename: {min_entropy_filename}")

    # Plot max and min entropy paintings
    visualize.display_painting(csv_path, max_entropy_painting_name, artist)
    visualize.display_painting(csv_path, min_entropy_painting_name, artist)


    #================================================================================
    # KL DIVERGENCE FOR ALL PAINTINGS
    #================================================================================
    
    print("\nWriting KL Divergence CSV...")
    probability_lib.kl_divergence_csv(csv_path, paintings, artist, artist_sorted_colors, artist_sorted_frequencies)
    print("KL Divergence CSV successfully written.")
    
    # Get max and min KL divergence paintings
    max_kl, max_kl_painting_name, max_kl_filename, min_kl, min_kl_painting_name, min_kl_filename = probability_lib.get_max_min_kl(kl_csv_path)
    
    print(f"\nHighest KL Divergence: {max_kl:.6f} bits")
    print(f"  Painting: {max_kl_painting_name}")
    print(f"  Filename: {max_kl_filename}")
    print(f"\nLowest KL Divergence: {min_kl:.6f} bits")
    print(f"  Painting: {min_kl_painting_name}")
    print(f"  Filename: {min_kl_filename}")

    # Plot max and min divergence paintings
    visualize.display_painting(csv_path, max_kl_painting_name, artist)
    visualize.display_painting(csv_path, min_kl_painting_name, artist)

    # # Plot KL Divergence points
    # visualize.plot_kl_points(kl_csv_path)


    #================================================================================
    # EXPERIMENTAL -- KL DIVERGENCE NORMAL DISTRIBUTION
    #================================================================================

    # Approximate a normal distribution from the KL divergences
    # kl_norm = probability_lib.calculate_kl_norm(kl_csv_path)

    # Plot KL norm distribution
    # visualize.plot_kl_norm(kl_norm, kl_csv_path)
    # visualize.plot_kl_norm(kl_norm)

    # print(kl_norm.cdf(2))
    # print(kl_norm.cdf(3))
    # print(kl_norm.cdf(3) - kl_norm.cdf(2))


    #================================================================================
    # EXPERIMENTAL -- KL DIVERGENCE GAMMA DISTRIBUTION
    #================================================================================

    # # Approximate a gamma distribution from the KL divergences
    # kl_gamma = probability_lib.calculate_kl_gamma(kl_csv_path)

    # visualize.plot_kl_gamma(kl_gamma)


#================================================================================
if __name__ == "__main__":
    main()