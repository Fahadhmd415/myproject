import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import time
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Covid","Pneumonia","Tuberculosis","About"],
        icons=["caret-right-fill","caret-right-fill","caret-right-fill","caret-right-fill"],
        default_index = 0

     )


if selected == "Covid":
    #covid model
    @st.cache(allow_output_mutation=True)
    def load_modelcovid():
        modelcovid = tf.keras.models.load_model("./Covid19_CNN_Classifier.h5")
        return modelcovid

    def import_and_predictcovid(image_data, model):
        new_img = np.array(image_data.convert('RGB'))  # our image is binary we have to convert it in array
        new_img = cv2.cvtColor(new_img, 1)  # 0 is original, 1 is grayscale
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # PX-Ray (Image) Preprocessing
        IMG_SIZE = (200, 200)
        img = cv2.equalizeHist(gray)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.  # Normalization

        # Image reshaping according to Tensorflow format
        X_Ray = img.reshape(1, 200, 200, 1)



        # Diagnosis (Prevision=Binary Classification)
        #diagnosis = model.predict_classes(X_Ray)
        diagnosis_proba = model.predict(X_Ray)
        probability_cov = diagnosis_proba * 100
        a = (probability_cov[0])
        a1 = a[0]
        probability_no_cov = (1 - diagnosis_proba) * 100
        b = (probability_no_cov[0])
        b1 = b[0]
        c = [int(a1),int(b1)]
        print(c)
        return c

    with st.spinner('Model is being loaded..'):
        modelcovid = load_modelcovid()

    st.title("""

          Detection Of Lung Diseases
    """)

    st.header("""
              Covid
             
             """

             )

    filecovid = st.file_uploader("Please upload an Xray scan file", type=["jpeg", "png","jpg"])
    st.set_option('deprecation.showfileUploaderEncoding', False)


    if filecovid is None:
       st.text("Please upload an image file")
    else:
        image = Image.open(filecovid)
        st.image(image, use_column_width=True)
        predictions = import_and_predictcovid(image, modelcovid)
        if st.button("Predict",key="covidbutton"):
            time.sleep(5)

            if predictions[0] >= 70:
                string = "This is Covid"
            else:
                string = " This is Normal"
            st.success(string)

if selected == "Pneumonia":
    #Pneumonia model
    st.title("""

          Detection Of Lung Diseases
    """)
    st.header("""
             Pneumonia
             """

             )


    @st.cache(allow_output_mutation=True)
    def load_modelpneumonia():
        return tf.keras.models.load_model("./modelPneumonia.h5")


    def make_predictionpneumonia(uploaded_image, model):
        uploaded_image.save("out.png")
        image = cv2.imread("out.png")
        image = cv2.resize(image, dsize=(200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image)
        image = image.reshape((1, image.shape[0], image.shape[1]))
        prediction = np.argmax(model.predict(image))
        print(prediction)

        return prediction

    filepneumonia = st.file_uploader("Please upload an Xray scan file", type=["jpeg", "png","jpg"],key= "Pneumonia")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    with st.spinner('Model is being loaded..'):
        modelpneumonia = load_modelpneumonia()


    if filepneumonia is None:
       st.text("Please upload an image file")
    else:
        image1 = Image.open(filepneumonia)
        st.image(image1, use_column_width=True)
        predictions = make_predictionpneumonia(image1, modelpneumonia)
        if st.button("Predict", key= "Pnemoniabutton"):
            time.sleep(5)
            if predictions == 0:
                string = "This is bacterial Pneumonia"
            elif predictions == 1:
                string = " This is Normal"
            else:
                string = "This is viral Pneumonia"
            st.success(string)
if selected == "Tuberculosis":
    #Tuberculosis model

    st.title("""

          Detection Of Lung Diseases
    """)
    st.header("""
             Tuberculosis
             """

             )







    @st.cache(allow_output_mutation=True)


    def load_modeltuberculosis():
        return tf.keras.models.load_model("./Tuberculosis.h5")




    def make_predictiontuberculosis(uploaded_image, model):
        size = (150, 150)
        image = ImageOps.fit(uploaded_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img[np.newaxis, ...]

        prediction = model.predict(img_reshape)
        return prediction

    filetuberculosis = st.file_uploader("Please upload an Xray scan file", type=["jpeg", "png","jpg"],key= "Tuberculosis")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    with st.spinner('Model is being loaded..'):
        modeltuberculosis = load_modeltuberculosis()


    if filetuberculosis  is None:
       st.text("Please upload an image file")
    else:
        image2 = Image.open(filetuberculosis )
        st.image(image2, use_column_width=True)
        predictions = make_predictiontuberculosis(image2, modeltuberculosis)
        if st.button("Predict", key= "Tuberculosisbutton"):
            time.sleep(5)
            if predictions == 0:
                string = "This is Tuberculosis"
            else:
                string = " This is Normal"
            st.success(string)
if selected == "About":
    st.title("Overview:")
    st.subheader("The main aim of this project is to get an instant result of an x-ray whether the person contains the following disease,")
    st.subheader("1. Covid")
    st.subheader("2. Pneumonia")
    st.subheader("3. Tuberculosis")
    st.subheader("This model is trained by using deep learning techniques such as the CNN(Convolutional Neural Network) algorithm and Xray of large datasets.")
    st.subheader("This is the final year project of an ECE students at Panimalar Engineering College from the batch 2018-2022." )
    st.subheader("This project was developed by")
    st.subheader("Aamir P")
    st.subheader("Dharvish RD")
    st.subheader("Fahadh Mohamed J")
    st.subheader("Joan Miracle J")
    st.subheader("Guide : Mrs.N.PRITHA, M.E.,"
              "  ASSISTANT PROFESSOR ")




