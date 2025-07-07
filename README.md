# Skin-Disease-Detection
A CNN-based deep learning model that detects skin diseases from images with high accuracy.

Project Title

Skin Disease Detection using CNN


 Description

A deep learning model that uses Convolutional Neural Networks to classify skin disease from image data. The system helps with early detection of conditions like eczema, acne, psoriasis, etc., by analyzing uploaded images.


Features

* Image classification using CNN
* Real-time disease prediction
* Trained on labeled dataset with high accuracy (\~90%)
* User-friendly interface (optional: Streamlit or Flask)
* Data augmentation for better generalization


 Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib / Seaborn
* (Optional) Flask / Streamlit

 
 Folder Structure

bash
Skin-Disease-Detection/
├── dataset/
├── model/
├── app.py / main.ipynb
├── requirements.txt
└── README.md


Installation

bash
git clone https://github.com/yourusername/Skin-Disease-Detection.git
cd Skin-Disease-Detection
pip install -r requirements.txt


How to Run

1. Train the model:

   bash
   python train.py
   

3. Run the web app (if any):

   bash
   streamlit run app.py
   

4. Or run the notebook:

   bash
   jupyter notebook main.ipynb
   

Results

* Accuracy: ~90% on test set
* Loss: include graph if needed
* Confusion Matrix: optional


 Sample Output
![Skin Disease Prediction](images/prediction_output.png)


Dataset
This project uses a publicly available skin disease image dataset from the internet for training and evaluation.

Dataset Name: HAM10000 / DermNet 

Description: The dataset contains high-resolution dermatoscopic images of skin lesions classified into different categories such as melanoma, benign nevi, and more.

Note: This dataset is used strictly for academic and research purposes.
Mention the dataset you used (e.g., HAM10000, DermNet, or a custom dataset) and link to it if it's public.

This project uses the HAM10000 dataset available on [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), which contains 10,000 dermatoscopic images of common pigmented skin lesions. The dataset is used solely for educational and research purposes in this project.

Future Work

* Expand dataset to include more diseases
* Mobile-friendly version
* Doctor feedback integration


 
