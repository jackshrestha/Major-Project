import numpy as np
import cv2
import csv
import os
import FaceEncoding as facee
import pickle
import sqlite3
import dlib
from os.path import exists
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import random

from keras.models import load_model
import imutils

import matplotlib
matplotlib.use("Agg")




cap=cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

def insertOrUpdate(First,Last,Roll,Batch,Faculty) :                                            
    connect = sqlite3.connect("Face-DataBase")    
    c=connect.cursor()                              
    cmd = "SELECT * FROM Students WHERE Roll = " + Roll                             
    cursor = c.execute(cmd)
    isRecordExist = 0
    for row in cursor:                                                          
        isRecordExist = 1
    if isRecordExist == 1:                                                      
        c.execute("UPDATE Students SET First = ? WHERE Roll = ?",(First, Roll))
        c.execute("UPDATE Students SET Last = ? WHERE Roll = ?",(Last, Roll))
        c.execute("UPDATE Students SET Batch = ? WHERE Roll = ?",(Batch, Roll))
        c.execute("UPDATE Students SET Faclty = ? WHERE Roll = ?",(Faculty, Roll))
    else:
        params = (First,Last,Roll,Batch,Faculty)                                             
        c.execute("INSERT INTO Students VALUES(?, ?, ? , ? , ?)", params)
    connect.commit()                                                            
    connect.close()   




def Registration():  

    First="Ishwor"
    Last="Shrestha"
    Roll = "73015"
    Faculty="BEL"
    Batch="2070"

    folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/"+Batch+"/"+Faculty+"/"+First+Last)

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    sampleNum = 0
    while(True):
        ret, img = cap.read() 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                      # reading the camera input                              # Converting to GrayScale
        dets = detector(img, 1)                                                    
        for i, d in enumerate(dets):                                                
            sampleNum += 1
            cv2.imwrite(folderPath + "/"+"Sample."+ str(sampleNum) + ".jpg",
                        gray[d.top()+10:d.bottom()+10, d.left()+10:d.right()+10])     
            cv2.rectangle(img, (d.left(), d.top())  ,(d.right(), d.bottom()),(0,255,0) ,2) 
                                                                 
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
        elif sampleNum>50:
            break

    insertOrUpdate(First,Last,Roll,Batch,Faculty)  
    cap.release()
    cv2.destroyAllWindows() 

    print("Training the model.........")

    dataset = folderPath

    EPOCHS = 30
    INIT_LR = 1e-3
    BS = 32
    IMAGE_DIMS = (96, 96, 3)

    data = []
    labels = []

    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images("dataset")))
    random.seed(42)
    random.shuffle(imagePaths)


    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)


    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.2, random_state=42)

   
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")
    model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    print("[INFO] training network...")
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)


    print("[INFO] serializing network...")
    model.save('facefeatures_model.h5')
   
    print("[INFO] serializing label binarizer...")
    f = open("labelbin", "wb")
    f.write(pickle.dumps(lb))
    f.close()



    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(folderPath+"/"+First)



#Registration()


def recognition():
    while True:
        ret, img = cap.read() 
        font = cv2.FONT_HERSHEY_PLAIN
        image = cv2.resize(img, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        output = img.copy()
        #print("[INFO] loading network...")
        model = load_model("facefeatures_model.h5")

        lb = pickle.loads(open("labelbin", "rb").read())
        #print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
                                        
        dets = detector(img, 1)

        for i, d in enumerate(dets):  
            #output = imutils.resize(output, width=400)
            cv2.rectangle(img, (d.left(), d.top())  ,(d.right(), d.bottom()),(0,255,0) ,2)    
           # cv2.putText(output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            #    1, (0, 255, 0), 2)
            #cv2.putText(img,"output",(d.left(),d.top()), font, 1,(255,255,255),2) 
            #print(output)
            label = "{}: {:.2f}%".format(label, proba[idx] * 100)
            output = imutils.resize(output, width=400)
            #cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
              #  0.7, (0, 255, 0), 2)
            cv2.putText(img, label,(d.left(),d.top()), font, 1,(255,0,0),1) 
            #print(output)
            print(label)
                                                                            
        cv2.imshow('frame', img)                                                   
        cv2.waitKey(1)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

recognition()


