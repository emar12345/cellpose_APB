import logging
from glob import glob
from pathlib import Path

from cellpose import models, io

logger = logging.getLogger(__name__)


# TODO: probably turn this into an actual class to be a true wrapper
# TODO: probably create a models_wrapper.py and move this there.

def run(images_directory, model_path, use_GPU):
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
    images = io.get_image_files(str(images_directory), "")
    channels = [0, 0]
    diam_labels = model.diam_labels.copy()
    for filename in images:
        img = io.imread(filename)
        masks, flows, styles = model.eval(img, diameter=diam_labels, channels=channels)
        # save results so you can load in gui
        io.masks_flows_to_seg(img, masks, flows, model.diam_labels, filename, channels)
        # save results as png
        io.save_masks(img, masks, flows, filename, png=False, tif=True)
