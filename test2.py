import cv2

# Pretrained classes in the model
classNames = {1: 'person', 17: 'cat', 18: 'dog'}

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

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
    model.setInput(cv2.dnn.blobFromImage(frame, size=(my_width, my_height), swapRB=True))
    output = model.forward()
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .7:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            #print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
            box_x = detection[3] * my_width
            box_y = detection[4] * my_height
            box_width = detection[5] * my_width
            box_height = detection[6] * my_height
            # send the picture to the server
            # ask the server to classify
            cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
            cv2.putText(frame, class_name, (int(box_x), int(box_y+.05*my_height)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

