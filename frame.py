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

my_width = 320
my_height = 240

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

# Loading model
model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, my_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, my_height)
camera.set(cv2.CAP_PROP_FPS, 40)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1);

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    model.setInput(cv2.dnn.blobFromImage(frame, size=(my_width, my_height), swapRB=True))
    output = model.forward()
    class_name = ''
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .7:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            print(class_name)
            box_x = detection[3] * my_width
            box_y = detection[4] * my_height
            box_width = detection[5] * my_width
            box_height = detection[6] * my_height
            cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
            cv2.putText(frame, class_name, (int(box_x), int(box_y+.05*my_height)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            if class_name == 'cat' or class_name == 'dog':
                break
            
    if class_name == 'cat' or class_name == 'dog':
        # save picture to the temp folder
        filetime = time.strftime("%Y%m%d%H%M%S")
        cv2.imwrite("temp/filename-%s.jpg" % filetime, frame)
        # send the picture to the server
        files = {'file': open("temp/filename-%s.jpg" % filetime, 'rb')}
        r = requests.post("http://35.244.89.241/newpendingpet.php", files=files)
        time.sleep(10)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

