from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six

from model.utils.bbox_tools import bbox_iou


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    prec, rec, tps, fps, fns = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap),
            'prec': prec, 'rec': rec,
            'tps': tps, 'fps': fps, 'fns': fns}


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    img_indices = iter([i for i in range(len(pred_bboxes))])
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)

    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    indices_tp = defaultdict(list)
    indices_fp = defaultdict(list)
    indices_fn = defaultdict(list)
    scores_tp = defaultdict(list)
    scores_fp = defaultdict(list)
    scores_fn = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult, img_idx in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults, img_indices):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                scores_fn[l].extend(pred_score_l)
                indices_fn[l].extend([img_idx] * len(pred_bbox_l))
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                scores_fp[l].extend(pred_score_l)
                indices_fp[l].extend([img_idx] * len(pred_bbox_l))
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)

            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for j, selected in enumerate(selec):
                if not selected and not gt_difficult_l[j]:
                    scores_fn[l].append(0)
                    indices_fn[l].append(img_idx)

            for i, gt_idx in enumerate(gt_index):
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                            scores_tp[l].append(pred_score_l[i])
                            indices_tp[l].append(img_idx)
                        else:
                            match[l].append(0)
                            scores_fp[l].append(pred_score_l[i])
                            indices_fp[l].append(img_idx)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
                    scores_fp[l].append(pred_score_l[i])
                    indices_fp[l].append(img_idx)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    tps = [None] * n_fg_class
    fps = [None] * n_fg_class
    fns = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        tps[l] = tp
        fps[l] = fp
        fns[l] = n_pos[l] - tp

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

        if l == 15:
            top_fp_indices = sorted(
                zip(scores_fp[l], indices_fp[l]), reverse=True)[:10]
            top_fn_indices = sorted(zip(scores_fn[l], indices_fn[l]))[
                :10]
            top_tp_indices = sorted(
                zip(scores_tp[l], indices_tp[l]), reverse=True)[:10]

            print(top_tp_indices)
            print(top_fp_indices)
            print(top_fn_indices)

    return prec, rec, tps, fps, fns


def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
