import cv2
import time
import requests

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

# Loading model
model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv2.CAP_PROP_FPS, 40)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    model.setInput(cv2.dnn.blobFromImage(frame, size=(320, 240), swapRB=True))
    output = model.forward()
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .8:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            print(class_name)
            if class_name == 'cat' or class_name == 'dog':
                # save picture to the temp folder
                filetime = time.strftime("%Y%m%d%H%M%S")
                cv2.imwrite("temp/filename-%s.jpg" % filetime, frame)
                # send the picture to the server
                files = {'file': open("temp/filename-%s.jpg" % filetime, 'rb')}
                r = requests.post("http://35.244.89.241/newpendingpet.php", files=files)
                print(r.text)
                time.sleep(1)
                camera.release()
                time.sleep(2)
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                camera.set(cv2.CAP_PROP_FPS, 40)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                time.sleep(1)
                break

camera.release()
