import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
path = 'D:\\-SASY-Smart-Attendance-System-main\\project\\images' #path for the folder where image is
images = []
personNames = []
myList = os.listdir(path)
print(myList) #taking all the images from the folder specified and return the name without the extension that is jpg,png,etc
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)

def faceEncodings(images): #for face recognition
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting the images that we have in bgr into rgb format
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name):  #for the attendence marking after the face is recognised and to store it in the csv file
    with open('C:\\Users\\ss586\\Documents\\project\\attendence.csv', 'r+') as f: #path for the csv file 
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')

encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(1) #zero for laptop cam 1 for external cam

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) 
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB) 

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]: # for the name to be displayed in the camera below the name of person
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13: # when we press enter the program will end
        break

cap.release()
cv2.destroyAllWindows()

#for conversion of csv file into excel file
df = pd.read_csv('C:\\Users\\ss586\\Documents\\project\\attendence.csv')  #path for csv file
print(df)
writer = pd.ExcelWriter('C:\\Users\\ss586\\Documents\\project\\attendence.xlsx') #path for excel file
df.to_excel(writer, index=False)
writer.save()