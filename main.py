# If you need to contact me in the future just send a request on github. (emar12345)
# I'm running this using the cellpose v2 docker container
# https://hub.docker.com/layers/biocontainers/cellpose/2.1.1_cv2/images/sha256-cfe36943a49590da85c64bb8006330397193de2732faad06f41260296e35978c?context=explore
# cellpose - 2.1.1_cv2


# basic parameters taken from cellpose collab page
# https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=ldNwr_zxMVha

from cellpose import core

from scripts.log_wrapper import IOWrapper
from scripts.run_wrapper import run
from scripts.test_wrapper import test, test_blanks
from scripts.train_wrapper import train
from pathlib import Path



## step 1: set up logging and check GPU
io_wrapper = IOWrapper()
logger, log_file, log_handler = io_wrapper.logger_setup(log_directory="./data/log/")

# check if you have GPU on, see thread if using pycharm
# https://stackoverflow.com/questions/59652992/pycharm-debugging-using-docker-with-gpus
use_GPU = core.use_gpu()
yn = ['NO', 'YES']
logger.info(f'>>> GPU activated? {yn[use_GPU]}')

## step 2: add all relevent directory paths.

# train dirs
train_set_0_2B_dir = Path('data/train/set_0_2B/')  # 28 IMAGES
train_set_0_B_dir = Path('data/train/set_0_B/')  # 34 IMAGES
train_set_1_dir = Path('data/train/set_1/')  # 115 IMAGES
train_set_full_dir = Path('data/train/set_full')  # 177 IMAGES
# test dirs
test_set_0_2B_dir = Path('data/test/test_8_set_0_2B/')  # 8 IMAGES
test_set_0_B_dir = Path('data/test/test_8_set_0_B/')  # 8 IMAGES
test_set_1_dir = Path('data/test/test_34_set_1/')  # 35 IMAGES
test_set_full_dir = Path('data/test/test_50_set_full/')  # 51 IMAGES
# empty dirs
empty_set_0_dir = Path('data/test/empty_set_0/')  # 21 IMAGES
empty_set_1_dir = Path('data/test/empty_set_1/')  # 12 IMAGES
# final train dirs, balanced somewhat around having an increase of 350 ROI per set
t_0 = Path('data/train/t_0/')  # 44 (8 from train_set_0_B, 8 from train_set_0_2B,28 from train_set_1)
t_1 = Path('data/train/t_1/')  # 88 (same pattern)
t_2 = Path('data/train/t_2/')  # 132 (same pattern)
t_3 = Path('data/train/t_3/')  # 177 (28 from train_set_0_B, 34 from train_set_0_2B, 115 from train_set_1)

"""
some options here:
1> train a model, or a set of models
2> test a model or a set of models on a single or multiple tests
3> label an image set with a specific model
Uncomment the section you want to work with. More information is in "scripts/*_wrapper.py"
"""

## 1> training a set of models
# training_list = [t_0, t_1, t_2]
# for train_dir in training_list:
#     model = train(train_dir=train_dir, model_type="CPx", use_GPU=True, n_epochs=1000, test_dir=test_set_full_dir)
#     model_path = io_wrapper.get_model_path(log_handler.logs)
#     io_wrapper.plot_training_stats(io_wrapper.get_training_stats(log_handler.logs), model_name=model_path.name)
#     del model, model_path

## 1> training single model
# model = train(train_dir=t_3, model_type="CPx", use_GPU=True, n_epochs=1000, test_dir=test_set_full_dir)
# model_path = io_wrapper.get_model_path(log_handler.logs)
# io_wrapper.plot_training_stats(io_wrapper.get_training_stats(log_handler.logs), model_name=model_path.name)


## 2> testing a set of models on several different tests
# model_gutter = Path('./data/model/gutter/t_3')
# models = sorted(list(model_gutter.glob('*')), key=lambda path: path.stat().st_mtime)
# for model_path in models:
#     test(test_dir=test_set_0_B_dir, model_path=model_path, use_GPU=True)
#     test(test_dir=test_set_0_2B_dir, model_path=model_path, use_GPU=True)
#     test(test_dir=test_set_1_dir, model_path=model_path, use_GPU=True)
#     test_blanks(test_dir=empty_set_0_dir, model_path=model_path, use_GPU=True)
#     test_blanks(test_dir=empty_set_1_dir, model_path=model_path, use_GPU=True)

## 3> labeling an image set using a given model
# model_path= 'data/model/gutter/t_0/cellpose_residual_on_style_on_concatenation_off_t_0_2023_08_08_19_46_32.613015_epoch_201'
# label_me = Path('data/label/label_test')
# run(label_me, model_path, use_GPU)

#todo: find where area of mask is (measure size) and use to output number of ROI and size of each ROI, or output to IMAGEJ. cellpose clearly has this function but I need to go find it.
