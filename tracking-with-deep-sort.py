import warnings
warnings.filterwarnings("ignore")

from models import *
from utils import *

import os, sys, time, datetime, random, cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# initialize deepSort object and video capture
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from utils import generate_detections
from deep_sort import preprocessing

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
# weights_path='config/download_weights.sh'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

#for detections
detection_model_path = '/root/tracking-with-sort/config/mars-small128.pb'
encoder = generate_detections.create_box_encoder(detection_model_path, batch_size=32)

def detect_image(frame):
    # scale and pad image
    img = Image.fromarray(frame)
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    detection_list = []
    for row in detections[0]:
        bbox, confidence = row[0:4], row[4]
        # import pdb; pdb.set_trace()
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        feature = encoder(bgr_image, [row[0:4].cpu()]).squeeze()
        if bbox[3] < 0: #min_height
            continue
        # import pdb; pdb.set_trace()
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

#for tracker
metric = nn_matching.NearestNeighborDistanceMetric(
    metric="cosine", matching_threshold=0.2, budget=None)
mot_tracker = Tracker(metric)


def detector(frame, min_confidence=0.6):
    # ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = frame
    detections = detect_image(pilimg)
    
    detections = [d for d in detections if d.confidence >= min_confidence]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, 1, scores)
    detections = [detections[i] for i in indices]
    
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = []
        mot_tracker.predict()
        import pdb; pdb.set_trace()
        mot_tracker.update(detections)
        for track in mot_tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            tracked_objects.append([
                bbox[0], bbox[1], bbox[2], bbox[3], track.track_id])

        for x1, y1, x2, y2, obj_id in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

    
import cv2
cap = cv2.VideoCapture('/root/tracking-with-sort/MOT16-04-raw.webm')
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
print("Processing Video...")
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    out.release()
    break
  output = detector(frame)
  out.write(output)
out.release()
print("Done processing video")