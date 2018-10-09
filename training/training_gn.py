# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.coco_dataset import COCODataset

checkpoint_dir=r"F:\Team-CV\checkpoints\torch_yolov3"
os.makedirs(checkpoint_dir, exist_ok=True)

TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained":"", #  set empty to disable
        # "backbone_pretrained":"../weights/darknet53_weights_pytorch.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "16,22, 22,42, 25,84, 29,67, 40,54, 44,37, 54,75, 74,127, 77,232",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.001,
        "other_lr": 0.001,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.2,
        "decay_step": 15,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 12,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"\\192.168.55.39\team-CV\dataset\origin_all_datas\all_scenes",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    "pretrain_snapshot": "",                        #  load checkpoint
    "evaluate_type": "",
    "try": 0,
    "export_onnx": False,
}

def train(config):
    config["global_step"] = config.get("start_step", 0)
    is_training = False if config.get("export_onnx") else True

    anchors = [int(x) for x in config["yolo"]["anchors"].split(",")]
    anchors = [[[anchors[i], anchors[i + 1]], [anchors[i + 2], anchors[i + 3]], [anchors[i + 4], anchors[i + 5]]] for i
               in range(0, len(anchors), 6)]
    anchors.reverse()
    config["yolo"]["anchors"] = []
    for i in range(3):
        config["yolo"]["anchors"].append(anchors[i])
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Optimizer and learning rate
    optimizer = _get_optimizer(config, net)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr"]["decay_step"],
        gamma=config["lr"]["decay_gamma"])

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)

    # Only export onnx
    # if config.get("export_onnx"):
        # real_model = net.module
        # real_model.eval()
        # dummy_input = torch.randn(8, 3, config["img_h"], config["img_w"]).cuda()
        # save_path = os.path.join(config["sub_working_dir"], "pytorch.onnx")
        # logging.info("Exporting onnx to {}".format(save_path))
        # torch.onnx.export(real_model, dummy_input, save_path, verbose=False)
        # logging.info("Done. Exiting now.")
        # sys.exit()

    # Evaluate interface
    # if config["evaluate_type"]:
        # logging.info("Using {} to evaluate model.".format(config["evaluate_type"]))
        # evaluate_func = importlib.import_module(config["evaluate_type"]).run_eval
        # config["online_net"] = net

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    dataloader = torch.utils.data.DataLoader(COCODataset(config["train_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=True,is_scene=True),
                                             batch_size=config["batch_size"],
                                             shuffle=True,drop_last=True, num_workers=0, pin_memory=True)

    # Start the training loop
    logging.info("Start training.")
    dataload_len=len(dataloader)
    for epoch in range(config["epochs"]):
        recall = 0
        mini_step = 0
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()
            config["global_step"] += 1
            for mini_batch in range(3):
                mini_step += 1
                # Forward and backward
                optimizer.zero_grad()
                outputs = net(images)
                losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls", "recall"]
                losses = [0] * len(losses_name)
                for i in range(3):
                    _loss_item = yolo_losses[i](outputs[i], labels)
                    for j, l in enumerate(_loss_item):
                        losses[j] += l
                # losses = [sum(l) for l in losses]
                loss = losses[0]
                loss.backward()
                optimizer.step()
                _loss = loss.item()
                # example_per_second = config["batch_size"] / duration
                # lr = optimizer.param_groups[0]['lr']

                strftime = datetime.datetime.now().strftime("%H:%M:%S")
                if (losses[7] / 3 >= recall / (step + 1)) or mini_batch == (3-1):#mini_batch为0走这里
                    recall += losses[7] / 3
                    print(
                        '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,avrec %.3f %d]' %
                        (strftime, epoch, config["epochs"], step, dataload_len,
                         losses[1], losses[2], losses[3],
                         losses[4], losses[5], losses[6],
                         _loss, losses[7] / 3, recall / (step + 1), mini_batch))
                    break
                else:
                    print(
                        '%s [Epoch %d/%d,batch %03d/%d loss:x %.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,prerc %.3f %d]' %
                        (strftime, epoch, config["epochs"], step, dataload_len,
                         losses[1], losses[2], losses[3],
                         losses[4], losses[5], losses[6],
                         _loss, losses[7] / 3, recall / step, mini_batch))
                    # logging.info(epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f "%
                    #     (epoch, step, _loss, example_per_second, lr))
                    # config["tensorboard_writer"].add_scalar("lr",
                    #                                         lr,
                    #                                         config["global_step"])
                    # config["tensorboard_writer"].add_scalar("example/sec",
                    #                                         example_per_second,
                    #                                         config["global_step"])
                    # for i, name in enumerate(losses_name):
                    #     value = _loss if i == 0 else losses[i]
                    #     config["tensorboard_writer"].add_scalar(name,
                    #                                             value,
                    #                                             config["global_step"])

        if (epoch % 2 == 0 and recall / len(dataloader) > 0.7) or recall / len(dataloader) > 0.96:
            torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, recall / len(dataloader), epoch))

        lr_scheduler.step()
    # net.train(True)
    logging.info("Bye bye")

def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
            {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
        ]
    else:
        logging.info("freeze backbone's parameters.")
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": logits_params, "lr": config["lr"]["other_lr"]},
        ]

    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        logging.info("Using SGD optimizer.")
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    config = TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Create sub_working_dir
    sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
        config['working_dir'], config['model_params']['backbone_name'], 
        config['img_w'], config['img_h'], config['try'],
        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(sub_working_dir):
        os.makedirs(sub_working_dir)
    config["sub_working_dir"] = sub_working_dir
    logging.info("sub working dir: %s" % sub_working_dir)

    # Creat tf_summary writer
    # config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
    # logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    train(config)

if __name__ == "__main__":
    main()
