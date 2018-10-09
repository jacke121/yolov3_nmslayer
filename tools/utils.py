import os
import torch
import random
import datetime
import numpy as np
from PIL import Image, ImageFont, ImageDraw
opj = os.path.join


def parse_cfg(cfgfile):
  """Parse a configuration file

  @Args
    cfgfile: (str) path to config file

  @Returns
    blocks: (list) list of blocks, with each block describes a block in the NN to be built
  """
  file = open(cfgfile, 'r')
  lines = file.read().split('\n')  # store the lines in a list
  lines = [x for x in lines if len(x) > 0]  # skip empty lines
  lines = [x for x in lines if x[0] != '#']  # skip comment
  lines = [x.rstrip().lstrip() for x in lines]
  file.close()

  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[":  # This marks the start of a new block
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block['type'] = line[1:-1].rstrip()
    else:
      key, value = line.split("=")
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks


def transform_coord(bbox, src='center', dst='corner'):
  """Transform bbox coordinates
    |---------|           (x1,y1) *---------|
    |         |                   |         |
    |  (x,y)  h                   |         |
    |         |                   |         |
    |____w____|                   |_________* (x2,y2)
       center                         corner

  @Args
    bbox: (Tensor) bbox with size [..., 4]

  @Returns
    bbox_transformed: (Tensor) bbox with size [..., 4]
  """
  flag = False
  if len(bbox.size()) == 1:
    bbox = bbox.unsqueeze(0)
    flag = True

  bbox_transformed = bbox.new(bbox.size())
  if src == 'center' and dst == 'corner':
    bbox_transformed[..., 0] = (bbox[..., 0] - bbox[..., 2]/2)
    bbox_transformed[..., 1] = (bbox[..., 1] - bbox[..., 3]/2)
    bbox_transformed[..., 2] = (bbox[..., 0] + bbox[..., 2]/2)
    bbox_transformed[..., 3] = (bbox[..., 1] + bbox[..., 3]/2)
  elif src == 'corner' and dst == 'center':
    bbox_transformed[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    bbox_transformed[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    bbox_transformed[..., 2] = bbox[..., 2] - bbox[..., 0]
    bbox_transformed[..., 3] = bbox[..., 3] + bbox[..., 1]
  else:
    raise Exception("format not supported! :shit:")

  if flag == True:
    bbox_transformed = bbox_transformed.squeeze(0)

  return bbox_transformed


def IoU(box1, box2, format='corner'):
  """Compute IoU between box1 and box2

  @Args
    box: (torch.cuda.Tensor) bboxes with size [# bboxes, 4]  # TODO: cpu
    format: (str) bbox format
      'corner' => [x1, y1, x2, y2]
      'center' => [xc, yc, w, h]
  """
  if format == 'center':
    box1 = transform_coord(box1)
    box2 = transform_coord(box2)

  # b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, :,0], box1[:,:, 1], box1[:,:, 2], box1[:,:, 3]
  # b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,:, 0], box2[:,:, 1], box2[:,:, 2], box2[:,:, 3]
  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:,2], box1[:, 3]
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:, 2], box2[:,3]

  inter_rect_x1 = torch.max(b1_x1, b2_x1)
  inter_rect_y1 = torch.max(b1_y1, b2_y1)
  inter_rect_x2 = torch.min(b1_x2, b2_x2)
  inter_rect_y2 = torch.min(b1_y2, b2_y2)

  inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
  b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

  return inter_area / (b1_area + b2_area - inter_area)





def get_current_time():
  """Get current datetime

  @Returns
    time: (str) time in format "dd-hh-mm"
  """
  time = str(datetime.datetime.now())
  time = time.split('-')[-1].split('.')[0]
  time = time.replace(' ', ':')
  day, hour, minute, _ = time.split(':')
  if day[-1] == '1':
    day += 'st'
  elif day[-1] == '2':
    day += 'nd'
  elif day[-1] == '3':
    day += 'rd'
  else:
    day += 'th'
  time = day + '.' + hour + '.' + minute
  return str(time)


def load_checkpoint(checkpoint_dir, epoch, iteration):
  """Load checkpoint from path

  @Args
    checkpoint_dir: (str) absolute path to checkpoint folder
    epoch: (int) epoch of checkpoint
    iteration: (int) iteration of checkpoint in one epoch

  @Returns
    start_epoch: (int)
    start_iteration: (int)
    state_dict: (dict) state of model
  """
  path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
  if not os.path.isfile(path):
    raise Exception("Checkpoint in epoch %d doesn't exist :sob:" % epoch)

  checkpoint = torch.load(path)
  start_epoch = checkpoint['epoch']
  state_dict = checkpoint['state_dict']
  start_iteration = checkpoint['iteration']

  assert iteration == start_iteration
  return start_epoch, start_iteration, state_dict


def save_checkpoint(checkpoint_dir, epoch, iteration, save_dict):
  """Save checkpoint to path

  @Args
    path: (str) absolute path to checkpoint folder
    epoch: (int) epoch of checkpoint file
    iteration: (int) iteration of checkpoint in one epoch
    save_dict: (dict) saving parameters dict
  """
  os.makedirs(checkpoint_dir, exist_ok=True)
  path = opj(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
  assert epoch == save_dict['epoch'], "`epoch` != save_dict's `start_epoch`"
  assert iteration == save_dict['iteration'], "`iteration` != save_dict's `start_iteration`"
  if os.path.isfile(path):
    print("Overwrite checkpoint in epoch %d, iteration %d :exclamation:" % (epoch, iteration))
  try:
    torch.save(save_dict, path)
  except Exception:
    raise Exception("Fail to save checkpoint")
  
  print("Checkpoint %s saved :heavy_check_mark:" % (str(epoch) + '.' + str(iteration) + '.ckpt'))


def log(writer, name, info, step):
  """Wrapper for tensorboard writer

  @Args
    writer: (SummaryWriter)
    name: (string) category name
    info: (dict or float) value
    step: (int) global steps
  """
  if isinstance(info, dict):
    for key, value in info.items():
      tag = name + '/' + key
      writer.add_scalar(tag, value, step)
  elif isinstance(info, float):
    writer.add_scalar(name, info, step)


def mAP(preds, gts, reso):
  """Compute mAP between prediction and ground truth

  @Args
    preds: (Tensor) with size [num_bboxes, 8=[batch idx, x1, y1, x2, y2, p0, conf, label]]
    gts: (Tensor) with size [bs, num_bboxes, 5=[xc, yc, w, h, label]]
    reso: (int) inputs resolution

  @Variables
    bs: (int) batch size
    nB: (int) number of boxes

  @Returns
    mAPs: (list)
  """
  mAPs = []

  for batch_idx, gt_batch in enumerate(gts):
    if gt_batch.size(0) == 0:
      continue
      
    correct = []
    detected = []
    gts=torch.Tensor().cuda()
    for i in range(gt_batch.size(0)):
      if torch.sum(gt_batch[i])>0:
        gts = gt_batch[i].unsqueeze(0) if gts.size(0) == 0 else torch.cat((gts, gt_batch[i].unsqueeze(0)), 0)

    gt_batch=gts
    gt_bboxes = transform_coord(gt_batch[:, 1:5]) * reso
    gt_labels = gt_batch[:, 0]

    try:
      pred_batch = preds[preds[..., 0] == batch_idx]
    except Exception:  # no prediction
      mAPs.append(0)
      break

    if pred_batch.size(0) == 0:
      correct.append(0)
      continue

    # sort pred by confidence
    _, indices = torch.sort(pred_batch[:, -2], descending=True)
    pred_batch = pred_batch[indices]

    for pred in pred_batch:
      pred_bbox = pred[1:5]
      pred_label = pred[-1]
      iou = IoU(pred_bbox.unsqueeze(0), gt_bboxes)
      _, indices = torch.sort(iou, descending=True)
      best_idx = indices[0]
      # FIXME: iou thresh as variblae (0.5)
      if iou[best_idx] > 0.5 and pred_label == gt_labels[best_idx] and best_idx not in detected:
        correct.append(1)
        detected.append(best_idx)
      else:
        correct.append(0)

    AP = ap_per_class(tp=correct, conf=pred_batch[:, -2], pred_cls=pred_batch[:, -1], target_cls=gt_labels)
    mAP = AP.mean()
    mAPs.append(mAP)
  if len(mAPs)==0:
     return 0
  else:
    return np.mean(mAPs)


def ap_per_class(tp, conf, pred_cls, target_cls):
  """ Compute the average precision, given the recall and precision curves.
  TODO: translate???
  Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
  # Arguments
      tp:    True positives (list).
      conf:  Objectness value from 0-1 (list).
      pred_cls: Predicted object classes (list).
      target_cls: True object classes (list).
  # Returns
      The average precision as computed in py-faster-rcnn.
  """

  # lists/pytorch to numpy
  tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf.cpu().data), np.array(pred_cls.cpu().data), np.array(target_cls)

  # Sort by objectness
  i = np.argsort(-conf)
  tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

  # Find unique classes
  unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

  # Create Precision-Recall curve and compute AP for each class
  ap = []
  for c in unique_classes:
    i = pred_cls == c
    n_gt = sum(target_cls == c)  # Number of ground truth objects
    n_p = sum(i)  # Number of predicted objects

    if (n_p == 0) and (n_gt == 0):
      continue
    elif (np == 0) and (n_gt > 0):
      ap.append(0)
    elif (n_p > 0) and (n_gt == 0):
      ap.append(0)
    else:
      # Accumulate FPs and TPs
      fpa = np.cumsum(1 - tp[i])
      tpa = np.cumsum(tp[i])

      # Recall
      recall = tpa / (n_gt + 1e-16)

      # Precision
      precision = tpa / (tpa + fpa)

      # AP from recall-precision curve
      ap.append(compute_ap(recall, precision))

  return np.array(ap)


def compute_ap(recall, precision):
  """Compute the average precision, given the recall and precision curves.
  TODO: translate
  Code originally from https://github.com/rbgirshick/py-faster-rcnn.
  # Arguments
      recall:    The recall curve (list).
      precision: The precision curve (list).
  # Returns
      The average precision as computed in py-faster-rcnn.
  """
  # correct AP calculation
  # first append sentinel values at the end

  mrec = np.concatenate(([0.], recall, [1.]))
  mpre = np.concatenate(([0.], precision, [0.]))

  # compute the precision envelope
  for i in range(mpre.size - 1, 0, -1):
    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

  # to calculate area under PR curve, look for points
  # where X axis (recall) changes value
  i = np.where(mrec[1:] != mrec[:-1])[0]

  # and sum (\Delta recall) * prec
  ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

if __name__ == '__main__':
    pass