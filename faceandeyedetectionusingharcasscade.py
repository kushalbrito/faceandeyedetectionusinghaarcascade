import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
def detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _,img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in face:
            cv2.rectangle(img, (x,y), (x+w , y+h), (255,0,0), 2)
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 2)
            for(ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex,ey), (ex+ew , ey+eh), (0,255,0), 2)
        cv2.imshow('facedetect', img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

detect()

