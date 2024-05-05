import matplotlib.pyplot as plt
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, preprocess
from torch.utils import data as data_
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from tqdm import tqdm
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import numpy as np
from torch.utils.data import Subset
import pickle
import pandas as pd

import torch as t
device = 'cuda' if t.cuda.is_available() else 'cpu'
print(device)

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


def combine_results():
    files = []
    for i in range(5):
        files.append(
            './detection_results_epoch_0/detection_results_{}.pkl'.format(i))

    combined_results = {
        "pred_bboxes": [],
        "pred_labels": [],
        "pred_scores": [],
        "gt_bboxes": [],
        "gt_labels": [],
        "gt_difficults": []
    }

    for file in files:
        with open(file, "rb") as f:
            results = pickle.load(f)
            for key in combined_results:
                combined_results[key].extend(results[key])

    with open("./detection_results/combined_detection_results.pkl", "wb") as f:
        pickle.dump(combined_results, f)

    eval_result = eval_detection_voc(
        combined_results["pred_bboxes"],
        combined_results["pred_labels"],
        combined_results["pred_scores"],
        combined_results["gt_bboxes"],
        combined_results["gt_labels"],
        combined_results["gt_difficults"],
        use_07_metric=True)

    return eval_result


def plot_precision_recall_per_class(result):
    plt.figure(figsize=(12, 8))
    for i, VOC_BBOX_LABEL_NAME in enumerate(VOC_BBOX_LABEL_NAMES):
        plt.plot(result['rec'][i], result['prec']
                 [i], label=VOC_BBOX_LABEL_NAME)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve by Class')
    plt.legend()
    plt.grid(True)
    plt.show()


# result = combine_results()

# data = {
#     'Class': [VOC_BBOX_LABEL_NAMES[i] for i in range(20)],
#     'TP': [result['tps'][i][-1] for i in range(20)],
#     'FP': [result['fps'][i][-1] for i in range(20)],
#     'FN': [result['fns'][i][-1] for i in range(20)],
#     'AP': [result['ap'][i] for i in range(20)],
# }

# df = pd.DataFrame(data)
# print(df)

# plot_precision_recall_per_class(result)


def eval_2007(dataloader, faster_rcnn, i, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [
                                                                       sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    with open("/content/drive/MyDrive/Colab Notebooks/fasterrcnn_vgg_upgrade/detection_results_{}.pkl".format(i), "wb") as f:
        pickle.dump({
            "pred_bboxes": pred_bboxes,
            "pred_labels": pred_labels,
            "pred_scores": pred_scores,
            "gt_bboxes": gt_bboxes,
            "gt_labels": gt_labels,
            "gt_difficults": gt_difficults
        }, f)

    print(i, "저장 완료")


def eval(**kwargs):
    opt._parse(kwargs)

    for i in range(5):
        testset = TestDataset(opt)
        total_samples = len(testset)
        indices = np.arange(total_samples)
        split_indices = np.array_split(indices, 5)
        split_indices = split_indices[i]
        testset = Subset(testset, split_indices)
        print(i, len(testset))
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False,
                                           pin_memory=True
                                           )

        faster_rcnn = FasterRCNNVGG16()
        trainer = FasterRCNNTrainer(faster_rcnn).to(device)
        trainer.load(
            "checkpoints_for_test/learning_rate_2e-3_best_epoch", parse_opt=True)
        eval_result = eval_2007(
            test_dataloader, faster_rcnn, i, test_num=opt.test_num)

# eval()
