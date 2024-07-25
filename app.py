import streamlit as st
import pickle
import time
import pandas as pd
from PIL import Image



# Sample data
data = pd.DataFrame({
    'Name': ['Image 1', 'Image 2', 'Image 3', 'Image 4'],
    'Image': ['images/data1.png', 'images/data2.png', 'images/data3.png', 'images/data4.png'],
    'Description': ['Use sentiment as hue to see the distribution of each numerical feature', 'pie chart', 'WordCloud', 'Plot 2x2 grid word cloud for each sentiment']
})

st.set_page_config (
    page_title="Twitter Sentiment Analysis",
    page_icon="images/icon.png",
)

function_option = st.sidebar.selectbox("Select the option: ",["Prediction", "Documentation"])
#st.sidebar.markdown("Developed By:")
#st.sidebar.markdown("1. Shyam Kumar\n2. Rakesh Kumar\n3. Sachin Kumar Verma")

if function_option == "Prediction":
    st.title("Twitter Sentiment Analysis")

    #load model
    model = pickle.load(open('twitter_sentiment.pkl','rb'))

    tweet = st.text_input("Enter Your tweet")

    submit = st.button('Predict')

    if submit:
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Prediction time taken: ',round(end-start, 2), 'seconds')
        print(prediction[0])
        st.write('Predicted Sentiment is: ',prediction[0])


else:
    # Function to load and display images
    def display_image(image_path, caption):
        image = Image.open(image_path)
        st.image(image, caption=caption)

    # Streamlit app
    st.title("Documentation for Twitter Sentiment Analysis")

    st.subheader("What is Sentiment Analysis?")
    st.markdown("The process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.")

    st.divider()

    image = Image.open('images/tsa.jpg')
    st.image(image)

    st.divider()

    st.subheader("Random Forest Classifier")
    st.markdown("Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.")

    st.subheader("Technologies Used")
    st.markdown("1. Anaconda\n2. Jupyter Notebook\n3. Python Libraries")

    st.subheader("Data Collection::Dataset")
    url1 = "https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/blob/master/twitter_sentiment.csv"
    url2 = "https://github.com/laxmimerit/preprocess_kgptalkie"
    st.markdown("check out on this [link](%s)" % url1)
    st.markdown("Preprocess KGP Talkies -> [link](%s)" % url2)

    st.subheader("Steps to Develop this Project:")
    st.markdown("1. Load the dataset\n2. Feature extraction\n3. Data visualization\n4. Data cleaning\n5. Train test split\n6. Model building\n7. Model training\n8. Model evaluation\n9. Model saving\n10. Streamlit application deploy")

    st.divider()

    # Display images and descriptions
    st.subheader("Images and Descriptions")
    for index, row in data.iterrows():
        st.subheader(row['Name'])
        st.write(row['Description'])
        display_image(row['Image'], row['Name'])
        st.divider()
    
    st.subheader("References")
    st.markdown("1. Preprocess KGP Talkies -> [link](%s)" % url2)




footer="""<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #F0F2F6;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
    <p><strong>&copy 2024. All Rights Reserved.</strong></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
