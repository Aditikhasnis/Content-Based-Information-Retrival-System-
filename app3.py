import streamlit as st
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the image features and filenames
feature_list = np.array(pickle.load(open('image_features_embedding.pkl', 'rb')))
filenames = pickle.load(open('img_files.pkl', 'rb'))

# Load the CSV file
csv_file_path = "D:/CBVIR/styles.csv"
df = pd.read_csv(csv_file_path)

# Define the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('CONTENT BASED IMAGE AND TEXT RETRIEVAL')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploader', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return distances.flatten(), indices.flatten()

def text_based_search(query, df, column='productDisplayName'):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df[column].astype(str))
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    return df.iloc[top_indices], cosine_similarities[top_indices]

# Specify the directory containing your images
images_directory = "D:/CBVIR/e-commerce/images"

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploader", uploaded_file.name), model)
        distances, indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(Image.open(os.path.join(images_directory, filenames[indices[0]])))
            st.write("Distance: {:.2f}".format(distances[0]))
        with col2:
            st.image(Image.open(os.path.join(images_directory, filenames[indices[1]])))
            st.write("Distance: {:.2f}".format(distances[1]))
        with col3:
            st.image(Image.open(os.path.join(images_directory, filenames[indices[2]])))
            st.write("Distance: {:.2f}".format(distances[2]))
        with col4:
            st.image(Image.open(os.path.join(images_directory, filenames[indices[3]])))
            st.write("Distance: {:.2f}".format(distances[3]))
        with col5:
            st.image(Image.open(os.path.join(images_directory, filenames[indices[4]])))
            st.write("Distance: {:.2f}".format(distances[4]))
    else:
        st.header("Some error occurred in file upload")

# Text-based search
search_query = st.text_input("Enter text to search")
if search_query:
    results, similarities = text_based_search(search_query, df)
    for i in range(len(results)):
        st.write(f"Result {i+1}: {results.iloc[i]['productDisplayName']}")
        st.write(f"Similarity: {similarities[i]:.2f}")
        image_path = os.path.join(images_directory, str(results.iloc[i]['id']) + ".jpg")  # Convert id to string before concatenation
        st.image(Image.open(image_path))
