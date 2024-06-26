import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def data(data_training):
        X = []
        Y = []
        for i in labels:
            path = data_training + "/" + i
            imgs = os.listdir(path)
            for j in imgs:
                path_img = path + "/" + j
                img = cv2.imread(path_img)
                img = cv2.resize(img,(200,200))
                X.append(img)
                Y.append(target[i])
        # return (yield X,Y)
        yield X
        yield Y


path_test = "D:/project_Lab_601/test_img"
file_path_test = os.listdir(path_test)

if (len(file_path_test) != 0):
    target = {"no_tumor": 0, "pituitary_tumor" : 1}
    labels = ["no_tumor","pituitary_tumor"]

    
    file_path = "D:/project_Lab_601/brain-tumor-detection/brain_tumor/Training" 
    # X,Y = 
    # X,Y = list(data(file_path))
    X,Y = list(data(file_path))
    # X = np.array)
    X = np.array(X)
    Y = np.array(Y)
    # print()
    # X = np.array(X)
    # Y = np.array(Y)
    np.unique(Y)
    cnt = pd.Series(Y).value_counts()

    X_updated = X.reshape(len(X), -1)
    # X_updated.shape

    xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)

    pca = PCA(.98)
    pca_train = xtrain
    pca_test = xtest

    sv = SVC()
    sv.fit(pca_train, ytrain)

    dec = {0:'No Tumor', 1:'Positive Tumor'}
    path_test_no_tumor = "D:/project_Lab_601/brain-tumor-detection/brain_tumor/Training/no_tumor"
    path_test_pituitary_tumor = "D:/project_Lab_601/brain-tumor-detection/brain_tumor/Training/pituitary_tumor"
    list_img_test1 = os.listdir(path_test_no_tumor)
    list_img_test2 = os.listdir(path_test_pituitary_tumor)

    img = cv2.imread(path_test_no_tumor + "/" + list_img_test1[0])    
    img1 = cv2.resize(img,(200,200))    
    img1 = img1.reshape(1,-1)

    img1 = cv2.imread(path_test + "/" + file_path_test[len(file_path_test) - 1])
    img1 = cv2.resize(img1,(200,200))
    img1 = img1.reshape(1,-1)
    p = sv.predict(img1)

    with open("predict.txt" , "w") as file:
        file.write(dec[p[0]])



