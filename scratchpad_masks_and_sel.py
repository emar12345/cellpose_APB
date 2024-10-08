import csv
import random
import shutil
from pathlib import Path

from cellpose import core

from scripts.log_wrapper import IOWrapper
from scripts.train_wrapper import get_masks

"""
Has stuff for two main processes.
1. reads images and associated npy file from an image set to get # of masks per image, outputs this a csv with format [(str)name, (float)mask]
2. creates a stratified sample set from an image set (based on # of masks), and moves these samples from the main image set.
2 (alt). I also included a random sample set creation option, but I would use stratified based on # of masks.
"""



def write_list_to_csv(data_list, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

def create_strata(data_list, num_samples):
    # Step 1: Identify distinct values for stratification
    distinct_strata = list(set(item[1] for item in data_list))
    # Step 2: Divide data_list into sublists based on stratification
    stratified_data = {stratum: [] for stratum in distinct_strata}
    for item in data_list:
        stratified_data[item[1]].append(item)
    # Step 3: Randomly select samples from each stratum
    stratified_sample = []
    sample_size_per_stratum = num_samples // len(distinct_strata)  # Adjust as needed
    for stratum, data_subset in stratified_data.items():
        if len(data_subset) >= sample_size_per_stratum:
            sampled_items = random.sample(data_subset, sample_size_per_stratum)
            stratified_sample.extend(sampled_items)
            data_subset = [item for item in data_subset if item not in sampled_items]
        else:
            stratified_sample.extend(data_subset)
    # If necessary, randomly select additional samples to meet the desired sample size
    remaining_samples = num_samples - len(stratified_sample)
    available_data = [item for item in data_list if item not in stratified_sample]
    stratified_sample.extend(random.sample(available_data, remaining_samples))
    return stratified_sample

def move_samples(stratified_sample, target_dir):
    for path, _ in stratified_sample:
        source_path = Path(path)
        target_path = target_dir / source_path.name
        shutil.move(source_path, target_path)
        source_path = source_path.parent / Path(source_path.stem + "_seg.npy")
        target_path = target_dir / source_path.name
        shutil.move(source_path, target_path)


io_wrapper = IOWrapper()
logger, log_file, log_handler = io_wrapper.logger_setup(log_directory="./data/log/")

# check if you have GPU on, see thread if using pycharm
# https://stackoverflow.com/questions/59652992/pycharm-debugging-using-docker-with-gpus
use_GPU = core.use_gpu()
yn = ['NO', 'YES']
logger.info(f'>>> GPU activated? {yn[use_GPU]}')

# train dirs
train_set_0_2B_dir = Path('data/train/set_0_2B/')  # 78 IMAGES
train_set_0_B_dir = Path('data/train/set_0_B/')  # 78 IMAGES

train_set_1_dir = Path('data/train/set_1/')  # 115 IMAGES
# test dirs
test_set_0_2B_dir = Path('data/test/test_8_set_0_2B/')  # 8 IMAGES
test_set_0_B_dir = Path('data/test/test_8_set_0_B/')  # 8 IMAGES
test_set_1_dir = Path('data/test/test_34_set_1/')  # 34 IMAGES
# empty dirs
empty_set_0_dir = Path('data/test/empty_set_0/')  # 21 IMAGES
empty_set_1_dir = Path('data/test/empty_set_1/')  # 12 IMAGES

# final testing sets
t_0 = Path('data/train/t_0/')
t_1 = Path('data/train/t_1/')
t_2 = Path('data/train/t_2/')
t_3 = Path('data/train/t_3')

list_of_dirs = [t_0, t_1, t_2,t_3]

for dir in list_of_dirs:
    nmasks = get_masks(dir, True)
    logger.info(nmasks)
    csv_filename = dir.name + ".csv"
    write_list_to_csv(nmasks, "data/mask/" + csv_filename)

# basic sample set creation. ideally you would stratify your samples.
# nmasks = get_masks(train_set_0_B_dir, True)
# random_sel = random.sample(nmasks, 8)
# print(random_sel)
#
# csv_filename = 'train_set_0_B_dir.csv'
# write_list_to_csv(nmasks, csv_filename)
# print(nmasks)


# strata example, could use scikit. DO NOT UNCOMMENT move_samples unless you want to move samples!

# nmasks = get_masks(train_set_0_2B_dir, True)
# logger.info(f'nmasks in {train_set_0_2B_dir}: {nmasks}')
# stratified_sample = create_strata(nmasks)
# logger.info(f'stratified sample of {train_set_0_2B_dir}: {stratified_sample}')
# # move_samples(stratified_sample, test_set_0_2B_dir)
# def the_pinnacle_of_repetition(directory):
#     nmasks = get_masks(train_set_0_B_dir, True)
#     logger.info(f'nmasks in {train_set_0_B_dir}: {nmasks}')
#     stratified_sample = create_strata(nmasks, 8)
#     logger.info(f'stratified sample of {train_set_0_B_dir}: {stratified_sample}')
#     move_samples(stratified_sample, directory)
#
#     nmasks = get_masks(train_set_0_2B_dir, True)
#     logger.info(f'nmasks in {train_set_0_2B_dir}: {nmasks}')
#     stratified_sample = create_strata(nmasks, 8)
#     logger.info(f'stratified sample of {train_set_0_2B_dir}: {stratified_sample}')
#     move_samples(stratified_sample, directory )
#
#     nmasks = get_masks(train_set_1_dir, True)
#     logger.info(f'nmasks in {train_set_1_dir}: {nmasks}')
#     stratified_sample = create_strata(nmasks, 28)
#     logger.info(f'stratified sample of {train_set_1_dir}: {stratified_sample}')
#     move_samples(stratified_sample, directory)
#
# the_pinnacle_of_repetition(t_0)
# the_pinnacle_of_repetition(t_1)
# the_pinnacle_of_repetition(t_2)
#

