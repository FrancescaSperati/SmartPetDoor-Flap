import cv2
import time
import requests

# Pretrained classes in the model
classNames = {17: 'cat', 18: 'dog'}

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
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    model.setInput(cv2.dnn.blobFromImage(frame, size=(320, 240), swapRB=True))
    output = model.forward()
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .7:
            class_id = detection[1]
            print(class_id)
            class_name=id_class_name(class_id,classNames)
            if class_name == 'cat' or class_name == 'dog':
                # save picture to the temp folder
                filetime = time.strftime("%Y%m%d%H%M%S")
                cv2.imwrite("temp/filename-%s.jpg" % filetime, frame)
                # send the picture to the server
                files = {'file': open("temp/filename-%s.jpg" % filetime, 'rb')}
                r = requests.post("http://35.244.89.241/newpendingpet.php", files=files)
                time.sleep(5)
                camera.release()
                time.sleep(8)
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                camera.set(cv2.CAP_PROP_FPS, 40)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(8)
                break

camera.release()
