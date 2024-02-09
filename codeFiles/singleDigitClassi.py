import cv2
import numpy as np
import keras

import cv2

loaded_model = keras.models.load_model('Trained_Model.h5')


h = 480
w = 640

cap = cv2.VideoCapture(0)
cap.set(3,w)
cap.set(4,h)

def preProcess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(28,28))
    img =  preProcess(img)
    img = img.reshape(1,28,28,1)
    img = img/255
    pred = loaded_model.predict(img)
    label = np.argmax(pred)
    prob = np.amax(pred)
    print(label)
    if prob > 0.6:
        cv2.putText(imgOriginal,str(label)+"    "+str(prob),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    cv2.imshow("Image", preProcess(imgOriginal))
    
cap.release()
cv2.destroyAllWindows()

