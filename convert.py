import csv

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as input_file:
        # Read lines from the input file
        lines = input_file.readlines()

    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Iterate through lines in the input file
        for line in lines:
            line = line.strip().split('|')

            # Print the line to check its content
            print(line)

            # Check if the line has the expected number of elements
            if len(line) >= 5:
                # Extract information from the line
                speaker = line[1].strip()
                emotion = line[2].strip()
                keyword = line[3].strip()
                dialogue = line[4].strip()

                # Write a row to the CSV file
                csv_writer.writerow([speaker, emotion, keyword, dialogue])
            else:
                print(f"Skipping line: {line}")

if __name__ == "__main__":
    convert_to_csv("data_combine_eng/all_data_pair.txt", "data_combine_eng/clause_keywords.csv")
