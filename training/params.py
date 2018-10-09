TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        # "backbone_pretrained": "../weights/darknet53_weights_pytorch.pth", #  set empty to disable
        "backbone_pretrained": "", #  set empty to disable
    },
    "yolo": {
        # "anchors": "15,22, 24,38, 25,64, 27,82, 39,58, 44,38, 62,77, 70,131, 78,233",
        # "anchors": "16,22, 22,42, 25,84, 28,67, 40,54, 44,37, 53,74, 70,207, 75,122",
        "anchors": "16,23, 23,38, 25,84, 30,65, 43,53, 47,37, 50,77, 60,118, 74,231",
        "classes": 1,
    },
    "lr": {
        "backbone_lr": 0.00001,
        "other_lr": 0.0001,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.1,
        "decay_step": 20,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 6,
    # "train_path": "../data/coco/trainvalno5k.txt",
    "train_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2train",
    # "train_path": r"\\192.168.55.73\Team-CV\dataset\origin_all_datas_0718back\_2train",
    "epochs": 2001,
    "img_h": 416,
    "img_w": 416,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    "working_dir": "YOUR_WORKING_DIR",              #  replace with your working dir
    # "pretrain_snapshot": r'',                        #  load checkpoint
    "pretrain_snapshot": r'E:\Team-CV\checkpoints\0723\YOLOv3_Pytorch/0.9450_0122.weights',                        #  load checkpoint
    "evaluate_type": "",
    "checkpoints":r"E:\Team-CV\checkpoints\0723\YOLOv3_Pytorch/",
    "try": 0,
    "export_onnx": False,
}
TESTING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_21",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": "16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236",
        "classes": 1,
    },
    "batch_size": 16,
    "confidence_threshold": 0.5,
    "classes_names_path": "../data/coco.names",
    "iou_thres": 0.5,
    "val_path": r"D:\data\VOC2007",
    "test_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2test/",
    "images_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2test\bj_800\JPEGImages/",
    "img_h": 352,
    "img_w": 352,
    "parallels": [0],
    "pretrain_snapshot": r"E:\Team-CV\checkpoints\0723\YOLOv3_pytorch/0.9132_0138.weights",
    'test_weights':r'F:\Team-CV\checkpoints\torch_yolov0824/'
}