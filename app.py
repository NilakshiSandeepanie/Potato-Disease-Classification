import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# Define remedies and consequences for each disease
disease_info = {
    'Potato__Early_Blight': {
        'consequences': ['Reduces potato yield', 'Affects quality of potatoes'],
        'remedies': ['Apply fungicide', 'Remove infected leaves', 'Improve air circulation']   
    },
    'Potato__Late_Blight': {
        'consequences': ['Significant reduction in potato yield', 'Loss of crop'],
        'remedies': ['Apply fungicide', 'Remove infected leaves', 'Practice crop rotation']      
    },
    'Potato__Healthy': {
        'consequences': ['No specific consequences'],
        'remedies': ['Maintain good crop management practices']       
    }
}

st.title('Potato Leaf Disease Prediction')

def main():
    file_uploaded = st.file_uploader('Choose an image...', type='jpg')
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence,  consequences, remedies = predict_class(image)
        st.write(f'<span style="font-size:20px; color:yellow">Prediction : {result}</span>', unsafe_allow_html=True)
        st.write(f'<span style="font-size:20px; color:yellow">Confidence : {confidence}%</span>', unsafe_allow_html=True)
        st.write('<span style="font-size:20px; color:yellow">Consequences : {}</span>'.format(', '.join(consequences)), unsafe_allow_html=True)
        st.write('<span style="font-size:20px; color:yellow">Remedies : {}</span>'.format(', '.join(remedies)), unsafe_allow_html=True)
        

def predict_class(image):
    with st.spinner('Loading Model...'):
        classifier_model = tf.keras.models.load_model('D:\\PotatoDisease\\potatoes.h5')

    shape = (256, 256, 3)
    model = tf.keras.Sequential([classifier_model])

    test_image = image.resize((256, 256))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    class_name = ['Potato__Early_Blight', 'Potato__Late_Blight', 'Potato__Healthy']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    
    # Get remedies and consequences for the predicted disease
    consequences = disease_info[final_pred]['consequences']
    remedies = disease_info[final_pred]['remedies']
    
    
    return final_pred, confidence, consequences, remedies

if __name__ == '__main__':
    main()

