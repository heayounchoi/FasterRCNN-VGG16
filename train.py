from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, preprocess
from torch.utils import data as data_
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from tqdm import tqdm
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from torchinfo import summary
import torch as t
import numpy as np
import wandb

# wandb.init(
# project="FasterRCNN",
# group="FasterRCNN",
# name="visualize_bird",
# notes="visualize_bird",
# )

device = 'cuda' if t.cuda.is_available() else 'cpu'
print(device)


def train(**kwargs):
    opt._parse(kwargs)
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    faster_rcnn = FasterRCNNVGG16()
    # print(summary(faster_rcnn, (1, 3, 600, 800)))
    # print(faster_rcnn)
    trainer = FasterRCNNTrainer(faster_rcnn).to(device)
    best_map = 0
    # lr_ = opt.lr
    # global_step = int(5011/opt.plot_every * 11)
    global_step = 0

    trainer.load("checkpoints_for_test/learning_rate_2e-3_best_epoch")
    # print("org_lr: ", trainer.faster_rcnn.optimizer.param_groups[0]['lr'])
    # trainer.faster_rcnn.scale_lr(opt.lr_decay)
    # print("new_lr: ", trainer.faster_rcnn.optimizer.param_groups[0]['lr'])

    for epoch in range(0, 1):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.to(device).float(
            ), bbox_.to(device), label_.to(device)
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                losses = trainer.get_meter_data()
                wandb.log({"rpn_loc_loss": losses["rpn_loc_loss"], "rpn_cls_loss": losses["rpn_cls_loss"],
                           "roi_loc_loss": losses["roi_loc_loss"], "roi_cls_loss": losses["roi_cls_loss"],
                           "total_loss": losses["total_loss"]}, step=global_step)
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))

                gt_img = np.transpose(gt_img, (1, 2, 0))
                wandb.log({"gt_img": [wandb.Image(gt_img)]}, step=global_step)

                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
                    [ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))

                pred_img = np.transpose(pred_img, (1, 2, 0))
                wandb.log(
                    {"pred_img": [wandb.Image(pred_img)]}, step=global_step)

                global_step += 1

        saved_path = trainer.save(save_path="checkpoints/", epoch=epoch)
        print("model saved at", saved_path)

        if epoch == 10:
            print("org_lr: ",
                  trainer.faster_rcnn.optimizer.param_groups[0]['lr'])
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            print("new_lr: ",
                  trainer.faster_rcnn.optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    train()
