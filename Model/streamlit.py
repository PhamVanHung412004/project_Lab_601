import streamlit as st
from PIL import Image 
import numpy as np
import os
import cv2

st.title("Using support vector machine for brain cancer classification problem")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
path_save = "D:/project_Lab_601/test_img/"
list_ = os.listdir(path_save)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='', use_column_width=True)
    img_test = np.array(image)
    cv2.imwrite(path_save + str(len(list_) + 1) + ".jpg",img_test)
    import model
    with open("predict.txt", "r") as file:
        data = file.read()
        if (data != ""):
            predict = "Predict: " + data
            st.write(predict)
    