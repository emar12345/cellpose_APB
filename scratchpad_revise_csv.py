
from pathlib import Path
import csv
source = Path('./data/fig/full_test_set/')

"""This was to get the data results from log_wrapper, written to csv files in data/fig/{something}
into a format for importation into the database (identifier.sql)"""

data = []
for path in source.rglob('*'):
    if path.suffix == '.csv' and "empty" in path.name:
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                model_name = path.parent.name
                test_name = path.stem
                x = [model_name, test_name]
                x.extend(row)
                data.append(x)

with open('empty_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for line in data:
        writer.writerow(line)

input_file = 'empty_data.csv'
output_file = 'empty_data_fixed.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header
    header = next(reader)
    writer.writerow(['model_name', 'epoch', 'test_name', 'bool'])

    for row in reader:
        model_name_parts = row[0].split('_epoch_')
        model_name = model_name_parts[0]
        epoch = model_name_parts[1]

        new_row = [model_name, epoch] + row[1:]
        writer.writerow(new_row)