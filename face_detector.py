import cv2

#load pre-trained data on frontal faces from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose webcam capture
webcam = cv2.VideoCapture(0)

#loop to iterate over frames
while True:
    #read current frame
    successful_frame_read, frame = webcam.read()

    #convert image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #show webcam
    cv2.imshow('Face Detector - Q to exit', frame)
    key = cv2.waitKey(1)

    #quit(q key)
    if key == 81 or key == 113:
        break

#release videocapture object
webcam.release()

