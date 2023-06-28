import pandas as pd
import pickle

def view_results(file_path):

    # Load data
    with open(file_path, 'rb') as f:
        setup, results = pickle.load(f)

    # Convert setup dictionary to a DataFrame for pretty printing
    setup_df = pd.DataFrame.from_dict(setup, orient='index', columns=['Value'])
    print("Setup:")
    print(setup_df)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Steps', 'Found goal'], index=range(1, len(results) + 1))
    print("\nResults:")
    print(results_df)

    # Calculate and print average steps
    avg_steps = results_df['Steps'].mean()
    print(f"\nAverage Steps: {avg_steps}")


# Example usage
# folder_name = 'experiment data/'
# data_file_name = 'experiment_data_20230628-140954.pickle'
# file_path = folder_name + data_file_name

# view_results(file_path)
