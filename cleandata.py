import csv

def filter_csv_to_training_data(input_file, output_file):
    training_data = []

    # Open the CSV file and read its contents
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Loop through each row in the CSV
        for row in csvreader:
            # Extract the 'text' column
            text = row['text'].strip()
            # Append the cleaned text to the training data
            training_data.append(text)
    
    # Save the filtered text into a new file, one sentence per line
    with open(output_file, mode='w', encoding='utf-8') as outfile:
        for line in training_data:
            outfile.write(line + '\n')

    print(f"Training data saved to {output_file}")

# Example usage
input_file = 'transcriptso.csv'
output_file = 'filtered_training_data.txt'
filter_csv_to_training_data(input_file, output_file)
