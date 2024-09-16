from flask import Flask 
import socketio
import eventlet
from keras.models import load_model
import base64 
from io import BytesIO
import cv2
from PIL import Image

sio = socketio.Server()

app = Flask(__name__)
speed_limit = 10
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(b64decode(data['image'])))
    # convert to numpy array 
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    send_control(steering_angle, 1.0)

@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

