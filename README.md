# 🫁 Lung Cancer Detection Using Deep Learning

## 📌 Project Overview
This project focuses on detecting lung cancer using deep learning techniques applied to medical imaging data. Early detection of lung cancer is critical for improving patient survival rates. This system uses a Convolutional Neural Network (CNN) model to classify lung scan images into three categories: benign, malignant, and normal. A Streamlit-based web interface enables users to upload images and receive real-time predictions with confidence scores.

---

## 🎯 Objectives
- Build an AI-based lung cancer detection system
- Apply deep learning for medical image classification
- Provide real-time prediction using a web interface
- Assist early diagnosis using AI

---

## ⚙️ Features
- Deep learning-based lung image classification
- Real-time prediction via Streamlit UI
- Image preprocessing and normalization
- Data augmentation for better accuracy
- Performance evaluation using confusion matrix
- Modular Python project structure

---

## 🧠 Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- PyTorch / TensorFlow
- OpenCV / PIL
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## 📂 Project Structure

SIC_Lung_Cancer_Detection/
│
├── app.py # Streamlit UI application  
├── predict.py # Prediction logic  
├── utils/  
│ └── preprocessing.py # Image preprocessing  
├── models/ # Model architecture files  
├── requirements.txt # Dependencies  
├── README.md # Documentation  
└── .gitignore # Ignored files  

---

## 🚀 How to Run the Project – Interview Explanation

1️⃣ Clone Repository
git clone https://github.com/Ajithv010/SIC_Lung_Cancer_Detection.git
cd SIC_Lung_Cancer_Detection

Explanation:
This step downloads the entire project from GitHub to the local system.
The 'git clone' command copies all project files, including source code,
requirements, and folder structure. After cloning, we move into the project
directory using the 'cd' command so we can run the project locally.

---

2️⃣ Install Requirements
pip install -r requirements.txt

Explanation:
The requirements.txt file contains all necessary Python libraries required
for the project such as deep learning frameworks, image processing libraries,
data handling tools, and Streamlit for the user interface. Installing these
ensures the project runs correctly without missing dependencies.

---

3️⃣ Run Application
streamlit run app.py

Explanation:
This command launches the Streamlit web application. It starts a local server
and opens the project interface in a browser. Users can upload lung scan images,
and the trained deep learning model processes the image to provide real-time
classification results such as benign, malignant, or normal along with confidence
scores.

---

Simple Interview Summary:
First, I clone the project from GitHub, install all required dependencies using
requirements.txt, and then run the Streamlit application, which provides a web
interface for uploading lung scan images and getting AI-based cancer predictions.
 

---

## 📊 Model Workflow
1. Upload lung image via Streamlit UI  
2. Image preprocessing and resizing  
3. Normalization and augmentation  
4. CNN model prediction  
5. Output classification with confidence score  

---

## 📈 Performance Evaluation
Model evaluation includes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix Visualization

These metrics help assess model performance and reliability.

---

## ⚠️ Note
Due to repository size limits, the following are excluded:
- Training datasets
- Training scripts
- Model weights
- Virtual environment files

They can be shared separately if required.

---

## 🔮 Future Enhancements
- Larger dataset training
- Model optimization
- Explainable AI integration
- Cloud deployment
- Clinical validation support

---

## 👨‍💻 Author
**Ajith V**  
Computer Science Engineering Student | AI/ML Enthusiast

---

## ⭐ Acknowledgment
This project demonstrates how artificial intelligence can assist healthcare by enabling faster and more accurate lung cancer detection using medical image analysis.
