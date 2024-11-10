import numpy as np
import cv2
import pickle

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

# SETUP THE VIDEO CAMERA
# url = "http://192.168.0.179:8080/video"  # Replace with the actual URL from the IP Webcam app
cap = cv2.VideoCapture(0)

# IMPORT THE TRAINED MODEL
pickle_in = open("model_trained.p", "rb")  # rb = READ BYTE
model = pickle.load(pickle_in)

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

while True:
    # READ IMAGE
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

    if probabilityValue > threshold:
        # DISPLAY CLASS NAME AND PROBABILITY
        cv2.putText(imgOriginal, f"CLASS: {getClassName(classIndex)}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()