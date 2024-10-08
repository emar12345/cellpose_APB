import logging
from pathlib import Path

from cellpose import io, models, dynamics
import numpy as np

logger = logging.getLogger(__name__)


# TODO: probably turn this into an actual class to be a true wrapper
# TODO: maybe move train and test into one models_wrapper.py?


def train(train_dir, use_GPU, n_epochs, min_train_masks=1, learning_rate=0.1,
          weight_decay=0.0001, pretrained_model=False, model_type=None, test_dir=None, model_name=None, save_path=None,
          nimg_per_epoch=None):
    """
    Wrapper for cellpose model.train(), I've pruned this a bit to only relevent params for this paper/experiment.
    :param train_dir: str
                what directory are training images and *.npy in



    :param use_GPU: bool

    :param n_epochs: int (default, 500)
                how many times to go through whole training set during training. From model.train()

    :param min_train_masks: int (default, 1)
                minimum number of masks an image must have to use in training set. From model.train()

    :param learning_rate: float or list/np.ndarray (default, 0.2)
                learning rate for training, if list, must be same length as n_epochs. From model.train()

    :param weight_decay: float (default, 0.00001)
                From model.train()

    :param pretrained_model: str or list of strings (optional, default False)
        full path to pretrained cellpose model(s), if None or False, no model loaded

    :param model_type: str (optional, default None)
        any model that is available in the GUI, use name in GUI e.g. 'livecell'
        (can be user-trained or model zoo)

    :param test_dir: str (default, None)
                What directory are testing images and *.npy in

    :param model_name: str (default, None)
                name of network, otherwise saved with name as params + training start time. From model.train()

    :param save_path: string (default, None)
            where to save trained model. If None, will be placed in models/train_dir/*

    :param nimg_per_epoch: int (optional, default None)
            minimum number of images to train on per epoch,
            with a small training set (< 8 images) it may help to set to 8. From to model.train()

    :return:
    """
    channels = [0, 0]
    if save_path is None:
        save_path = Path('data/model/' + train_dir.name)
    # check params
    run_str = f'python -m cellpose --use_gpu --verbose --train --dir {train_dir} --pretrained_model {pretrained_model} --model_type {model_type} --chan {channels[0]} --chan2 {channels[1]} --n_epochs {n_epochs} --learning_rate {learning_rate} --weight_decay {weight_decay}'
    if test_dir is not None:
        run_str += f' --test_dir {test_dir}'
    run_str += ' --mask_filter _seg.npy'
    print(run_str)

    # actually start training

    # DEFINE CELLPOSE MODEL (without size model), probably put this to a separate function, to be able to access from main
    if model_type and pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, model_type=model_type, pretrained_model=pretrained_model)
        logger.info(f'model type {model_type} and pretrained model {pretrained_model}')
    if model_type and not pretrained_model:
        model = models.CellposeModel(gpu=use_GPU, model_type=model_type)
        logger.info(f'model type {model_type} only')
    if pretrained_model and not model_type:
        model = models.CellposeModel(gpu=use_GPU, pretrained_model=pretrained_model)
        logger.info(f'pretrained model {pretrained_model} only')

    # get files
    output = io.load_train_test_data(str(train_dir), str(test_dir), mask_filter='_seg.npy')
    train_data, train_labels, _, test_data, test_labels, _ = output
    model.train(train_data, train_labels,
                test_data=test_data,
                test_labels=test_labels,
                channels=channels,
                save_path=save_path,
                save_every=10,
                save_each=True,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                nimg_per_epoch=nimg_per_epoch,
                SGD=True,
                min_train_masks=min_train_masks,
                model_name=model_name)
    return model


def get_masks(directory, use_GPU):
    output = io.load_train_test_data(str(directory), mask_filter='_seg.npy')
    model = models.CellposeModel(gpu=use_GPU)
    train_data, train_labels, train_paths, test_data, test_labels, test_paths = output
    train_flows = dynamics.labels_to_flows(train_labels, files=None, use_gpu=model.gpu, device=model.device)
    nmasks = np.array([label[0].max() for label in train_flows])
    return list(zip(train_paths, nmasks))
