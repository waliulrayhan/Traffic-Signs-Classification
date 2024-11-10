import numpy as np
import cv2
import pickle
from flask import Flask, render_template, Response, jsonify

#############################################

app = Flask(__name__)

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

# IMPORT THE TRAINED MODEL
model_path = "model_trained.p"  # Updated to load from provided model path
pickle_in = open(model_path, "rb")  # rb = READ BYTE
model = pickle.load(pickle_in)

# Global variables to store the latest class name and probability
latest_class_name = "Unknown"
latest_probability = 0.0

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize to [0, 1]
    return img

def getClassName(classNo):
    classNames = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h', 3: 'Speed Limit 60 km/h',
        4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h', 6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h', 9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road',
        24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
        29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
        33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
        38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classNames.get(classNo, "Unknown")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global latest_class_name, latest_probability  # Use global variables
    cap = cv2.VideoCapture(0)
    while True:
        success, imgOriginal = cap.read()
        if not success:
            break

        # PROCESS IMAGE
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)

        # Update latest class name and probability
        if probabilityValue > threshold:
            latest_class_name = getClassName(classIndex)
            latest_probability = round(probabilityValue * 100, 2)
        else:
            latest_class_name = "Unknown"  # or "N/A"
            latest_probability = 0.0

        ret, buffer = cv2.imencode('.jpg', imgOriginal)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_info')
def latest_info():
    return jsonify(className=latest_class_name, probability=float(latest_probability))  # Convert to standard float

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
