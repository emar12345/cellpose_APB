import csv
import logging
import warnings
from pathlib import Path

import numpy as np
from cellpose import io, models
from cellpose.metrics import _intersection_over_union, _true_positive

logger = logging.getLogger(__name__)


# TODO:probably turn this into an actual class to be a true wrapper
# TODO: probably merge with train_wrapper.py as models_wrapper.py?


# warnings.filterwarnings("error")


# modify this script to return array of > 0.5? it is complicated because we want to know 'failed' predictions, but we
# can't exactly match them up. assume IOU array is AxB, Find the highest value in each A row. This would skip the cost
# matrix, but give a somewhat useful metric. The cost matrix would tell us if the predicted mask was bleeding over into
# another region. As in, we discover IOU[3[1]] = 0.5, so we would have output[3] = 0.5, however we have some false
# positive as IOU[3[2]] = 0.25, which wouldn't be recorded. perhaps just return the linear sum?
# def _true_positive(iou, th):
#     """ true positive at threshold th
#
#     Parameters
#     ------------
#
#     iou: float, ND-array
#         array of IOU pairs
#     th: float
#         threshold on IOU for positive label
#
#     Returns
#     ------------
#
#     tp: float
#         number of true positives at threshold
#
#     ------------
#     How it works:
#         (1) Find minimum number of masks
#         (2) Define cost matrix; for a given threshold, each element is negative
#             the higher the IoU is (perfect IoU is 1, worst is 0). The second term
#             gets more negative with higher IoU, but less negative with greater
#             n_min (but that's a constant...)
#         (3) Solve the linear sum assignment problem. The costs array defines the cost
#             of matching a true label with a predicted label, so the problem is to
#             find the set of pairings that minimizes this cost. The scipy.optimize
#             function gives the ordered lists of corresponding true and predicted labels.
#         (4) Extract the IoUs fro these parings and then threshold to get a boolean array
#             whose sum is the number of true positives that is returned.
#
#     """
#     n_min = min(iou.shape[0], iou.shape[1])
#     costs = -(iou >= th).astype(float) - iou / (2 * n_min)
#     true_ind, pred_ind = linear_sum_assignment(costs)
#     match_ok = iou[true_ind, pred_ind] >= th
#     tp = match_ok.sum()
#     return tp
def average_precision(masks_true, masks_pred, filename, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------

    masks_true: list of ND-arrays (int) or ND-array (int)
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int)
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError('metrics.average_precision requires len(masks_true)==len(masks_pred)')

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    # n_true[n] = number of true masks in a given image.
    # mask_true = assigned value of each pixel in image.
    for n in range(len(masks_true)):
        # _,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            # dr ross wants to see this. basically the IOU matrix of true x pred, and then _truepositive finds the matches.
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        # try:
        #     ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        # except RuntimeWarning as e:
        #     print(f"n:{n}, tp[n]: {tp[n]}, fp[n]: {fp[n]}, fn[n]: {fn[n]}, e: {e}")
        #     print(tp[n], fp[n], fn[n], e)
        #     ap[n] = 1

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def test(test_dir, model_path, use_GPU):
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=str(model_path))
    channels = [[0, 0]]
    diam_labels = model.diam_labels.copy()
    # get files (during training, test_data is transformed, so we will load it again)
    output = io.load_train_test_data(str(test_dir), mask_filter='_seg.npy')

    model_name = Path(model_path).name
    test_dir_name = Path(test_dir).name
    num_img = len(output[2])
    logger.info(f'>>> testing model "{model_name}" on "{test_dir_name}" dataset containing {num_img} images.')

    test_data, test_labels = output[:2]
    # run model on test images
    masks = model.eval(test_data,
                       channels=channels,
                       diameter=diam_labels)[0]

    # check performance using ground truth labels
    ap, tp, fp, fn = average_precision(test_labels, masks, output[2])
    # IOU for individual images at threshold 0.5, ap[:,1] would be for threshold 0.75. To understand the metrics/conclusion, look at 'average_precision()'.
    logger.info(f'>>> precision at iou threshold 0.50: {list(zip(output[2], ap[:, 0]))}')
    logger.info(f'>>> precision at iou threshold 0.75: {list(zip(output[2], ap[:, 1]))}')
    iou_50 = np.mean(ap[:, 0])
    iou_75 = np.mean(ap[:, 1])
    row_names = ["NAME", "threshold 0.50", "threshold 0.75", "threshold 0.90"]
    stats = [list(row) for row in zip(output[2], ap[:, 0], ap[:, 1], ap[:, 2])]
    write_csv(stats, model_name, test_dir_name + '_AP_stats', row_names)

    # optionally we could get tp, fp, fn? e.g:
    row_names = ["NAME", "AP T 0.50", "TP T 0.50", "FP T 0.50", "FN T 0.50"]
    stats = [list(row) for row in zip(output[2], ap[:, 0], tp[:, 0], fp[:, 0], fn[:, 0])]
    write_csv(stats, model_name, test_dir_name + '_050_stats', row_names)

    logger.info(
        f'>>> average precision at iou threshold 0.50 = {iou_50:.3f}, average precision at iou threshold 0.75 = {iou_75:.3f}.')

    return stats


def write_csv(stats, model_name, test_name, row_names):
    # todo: combine this with the one from log_wrapper. trying to get this out the door, but in the future someone should rewrite this.
    path = Path("./data/fig", model_name)
    path.mkdir(parents=True, exist_ok=True)
    csv_file = path / Path(test_name + ".csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_names)
        writer.writerows(stats)


def test_blanks(test_dir, model_path, use_GPU):
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=str(model_path))
    channels = [[0, 0]]
    diam_labels = model.diam_labels.copy()
    # get files (during training, test_data is transformed, so we will load it again)
    output = io.load_train_test_data(str(test_dir), mask_filter='_seg.npy')

    model_name = Path(model_path).name
    test_dir_name = Path(test_dir).name
    num_img = len(output[2])
    logger.info(f'>>> testing model "{model_name}" on "{test_dir_name}" dataset containing {num_img} images.')

    test_data, test_labels = output[:2]
    # run model on test images
    masks = model.eval(test_data,
                       channels=channels,
                       diameter=diam_labels)[0]

    # check performance using ground truth labels
    ap = average_precision(test_labels, masks, output[2])[0]
    nans_50 = np.count_nonzero(np.isnan(ap[:, 0]))
    row_names = ["NAME", "BOOL"]
    stats = [list(row) for row in zip(output[2], np.isnan(ap[:, 0]))]
    write_csv(stats, model_name, test_dir_name + '_stats', row_names)
    logger.info(f'{list(zip(output[2], ap[:, 0]))}')
    logger.info(
        f'>>> {nans_50} out of {num_img} blanks predicted correctly. Percent: {nans_50 / num_img}')
    return nans_50 / num_img


def test_multiple(model_dir, test_dir, use_GPU):
    path = Path(model_dir)
    models = path.glob('*')
    for model in models:
        if model.is_file():
            print(str(model))
            test(test_dir, str(model), use_GPU)
