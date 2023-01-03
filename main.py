import streamlit as st
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import cv2 

def face_detect(img):
    # creating image array so that openCV functions can be used over it
    image_array = np.array(img)
    # Detecting bounding boxes using MTCNN model
    mtcnn_model = MTCNN(keep_all = True)
    bounding_boxes,confidence,landmarks = mtcnn_model.detect(image_array,landmarks = True)
    # extracting face using bounding box coordinates 
    face_array = []
    for i in range(len(bounding_boxes)):
        x1,y1,x2,y2 = bounding_boxes[i]
        face_array = image_array[int(y1):int(y2),int(x1):int(x2)]
    return Image.fromarray(face_array)


if __name__ == "__main__":
    # app description
    st.title('App to :red[Detect] :blue[Faces]')
    
    full_image = Image.open('/Users/yash/Desktop/face_detection_app/face.png') 
    face_image = Image.open('/Users/yash/Desktop/face_detection_app/full.png')
    col1,col2 = st.columns(2)
    with col1:
        st.subheader('Full image')
        st.image(full_image)
    with col2:
        st.subheader('Face detected from image')
        st.image(face_image)

    ## Getting image as input from user
    st.subheader('Upload image to detect faces from')
    user_img = st.file_uploader('Chose image from which you want to detect faces', type=['jpg','png','jpeg'])

    ## Detecting face and displaying results
    if user_img:
        col3,col4 = st.columns(2)
        user_img = Image.open(user_img)
        with col3:
            st.subheader('Uploaded image')
            st.image(user_img)
        with col4:
            st.subheader('Face detected')
            st.image(face_detect(user_img).convert('RGB'))