input_file_path = "data_combine_eng/train.txt"
output_file_path = "data_combine_eng/train.txt"

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    content = input_file.read()

# Replace "|" with ","
modified_content = content.replace(' | ', ',')

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(modified_content)

print(f"Conversion completed. Modified content saved to {output_file_path}")
