import RPi.GPIO as GPIO
import time

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz

# OPEN
p.start(0) # Initialization
p.ChangeDutyCycle(8)
#time.sleep(0.4)
#p.stop()

time.sleep(5)

# CLOSE
#p.start(23)
p.ChangeDutyCycle(12)
time.sleep(0.5)
p.stop()

GPIO.cleanup()

