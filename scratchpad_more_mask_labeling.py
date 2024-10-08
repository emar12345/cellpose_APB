import csv
"""
literally just reads some_list_of_masks.csv with format [(str)name, (float/int)masks]
and gets what original group they were part of because I didn't document it when I initially seperated them
into the test sets (from scratchpad_masks_and_sel.py)
"""
# Assuming the data is stored in a file called "t_0.csv"
filename = "empty_set_0.csv"

# Initialize a variable to hold the total count of apb
B3_apb = 0
B1_apb = 0
B2_apb = 0
# Open the CSV file
with open(filename, 'r') as file:
    reader = csv.reader(file)
    name_list = []
    # Iterate through the rows
    for row in reader:
        # Assuming name is in the first column and apb is in the second
        name, apb = row
        name_list.append(name)
        # If the name contains "OutlinedJB", add the apb value to the total count
        if "OutlinedJB" in name:
            B3_apb += float(apb)
        if "2B" in name and "OutlinedJB" not in name:
            B1_apb += float(apb)
        if "2B" not in name and "OutlinedJB" not in name:
            B2_apb += float(apb)

# Print the total count of apb
print(f"Total number of apb (with '2B'): {B1_apb}")
print(f"Total number of apb (with 'B' only): {B2_apb}")
print(f"Total number of apb (with 'OutlinedJB'): {B3_apb}")
print(f"total number of apb {(B1_apb+B2_apb+B3_apb)}")
print(f'total number of images, {len(name_list)}')


