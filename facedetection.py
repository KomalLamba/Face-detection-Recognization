import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_face.xml')
'''
#for image
img = cv.imread('group 2.jpg')
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    #cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    face_roi = gray_image[y:y+h, x:x+w]
    blurred_face = cv.blur(face_roi, (30, 30))
    img[y:y+h, x:x+w] = cv.cvtColor(blurred_face, cv.COLOR_GRAY2BGR)

output_image = cv.imwrite('output_image.png', img)
'''
# video(img is frame)



video_capture = cv.VideoCapture('multiperson2.mp4')

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output3.mp4', fourcc, 30.0,(int(video_capture.get(3)), int(video_capture.get(4))))
while True:  
    ret, frame = video_capture.read() 

    if not ret: break  

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
    for (x, y, w, h) in faces: 
        face_roi = frame[y:y+h, x:x+w] 
        blurred_face = cv.blur(face_roi, (30, 30)) 
        frame[y:y+h, x:x+w] = blurred_face 
            #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
            
        out.write(frame)
        cv.imshow('Video Stream with Face Detection', frame) 
    if cv.waitKey(25) & 0xFF == ord('q'): 
            break

    #video_capture.release() 
out.release()
cv.destroyAllWindows() 


