# coding='utf-8'
import datetime
import os
import numpy as np
import logging
import shutil
import cv2
import torch
import torch.nn as nn
from tools import savexml
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLayer
from common.utils import non_max_suppression, bbox_iou

TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_21",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236",
        "classes": 1,
    },
    "batch_size": 1,
    "confidence_threshold": 0.8,
    "iou_thres": 0.3,
    # "images_path":  r"\\192.168.55.38\Team-CV\cam2pick\camera_pic_0711\sh_wuding\rec_pic/",
    "images_path":  r"\\192.168.55.73\Team-CV\dataset\origin_all_datas\_2test\sh_wd\JPEGImages/",
    # "images_path":  r"\\192.168.25.20\SBDFileSharing\Team-CV\find_mouse\background\huaping\pic/",
    # "images_path":  r"E:\github\YOLOv3_PyTorch\evaluate\test_paste/",
    "img_h": 352,
    "img_w": 352,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
    "pretrain_snapshot": r"F:\Team-CV\checkpoints\torch_yolov0804\0.9654_1667.weights",
    # "pretrain_snapshot": r"F:\Team-CV\checkpoints\torch_yolov3\0.9213_0090.weights",
    # "pretrain_snapshot": r"E:\github\YOLOv3_PyTorch\evaluate/weights/0.9296_0026.weights",
}

def test(config):
    is_training = False
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

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLayer(config["batch_size"],i,config["yolo"]["anchors"][i],
                                     config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_path = os.listdir(config["images_path"])
    # images_path = [os.path.join(config["images_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start inference
    batch_size = config["batch_size"]
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            if not path.endswith(".jpg") and (not path.endswith(".png")) and not path.endswith(".JPEG"):
                continue
            logging.info("processing: {}".format(path))
            image = cv2.imread(os.path.join(config["images_path"], path), cv2.IMREAD_COLOR)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            time1=datetime.datetime.now()
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"])
            print("time",(datetime.datetime.now()-time1).microseconds)

        # write result images. Draw bounding boxes and labels of detections
        for idx, detections in enumerate(batch_detections):
            # image_show =images_origin[idx]
            if detections is not None:

                anno = savexml.GEN_Annotations(path)
                anno.set_size(1280, 720, 3)

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config["img_h"], config["img_w"]
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    # image_show = cv2.rectangle(images_origin[idx], (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 1)
                    anno.add_pic_attr("mouse", int(x1.cpu().data), int(y1.cpu().data), int(box_w.cpu().data) , int(box_h.cpu().data) ,"0")

                print("output/"+str(step)+".jpg")
                anno.savefile("Annotations/"+ path[:-4] + '.xml')
                cv2.imwrite("output/"+os.path.basename(path),images_origin[idx])
            # cv2.imshow('1', image_show)
            # cv2.waitKey()
    logging.info("Save all results to ./output/")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    os.makedirs('Annotations', exist_ok=True)
    logging.basicConfig(level=logging.DEBUG,format="[%(asctime)s] %(message)s")

    config =TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)
