Smart Pet Door

2019 - Advanced Studio Project @ AIT 6431 Francesca Sperati - 6590 Charlie Chiu

To connect to the pi from terminal: ssh pi@10.1.7.16 

Into the raspberry pi there are two main scripts:

- openclose.py
  connects to the servomotor, moves the mechanical arm, waits 5 seconds, then moves it back to the original position
 - petdetect.py
  operates object detection from TensorFlow using the MobileNet dataset. 
  If a 'cat' or 'dog' is detected, it will take a picture and send it to the Server


All code of this repository is property and copyright of Francesca Sperati and Charlie Chiu, 
Licence is proprietary and all rights are reserved
