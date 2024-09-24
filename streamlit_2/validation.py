import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
st.title("Face Recognition Borwita")
upload=st.file_uploader('Upload Image here',['jpeg','png','jpg'])
if upload :
    # Load a model
    model = YOLO(r"streamlit_2/best.pt")  # pretrained YOLOv8n model
    im=Image.open(upload)
    # Run batched inference on a list of images
    results = model.predict([im],conf=0.55)  # return a list of Results objects

    # Process results list
    for result in results:
        result.save(r'streamlit_2/result.jpg')
        st.image(r'streamlit_2/result.jpg')
        for box in result.boxes:  # Loop through each detected object
            class_id = int(box.cls)  # Get the class ID
            confidence = box.conf  # Get the confidence score
            # name = model.names[class_id]  # Get the class name
            st.text(f'Name: {result.names[class_id]},conf: {confidence[0]} ')
    
    

    
