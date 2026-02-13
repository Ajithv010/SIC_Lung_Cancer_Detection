# Lung Cancer Detection AI Project

## Folder Structure

lung_cancer_detection/
│
├── data/
│   └── The IQ-OTHNCCD lung cancer dataset/
│       ├── Benign cases/
│       ├── Malignant cases/
│       └── Normal cases/
│
├── models/
├── utils/
│   └── preprocessing.py
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md

## How to Run

1. Install requirements:
   pip install -r requirements.txt

2. Add dataset images inside:
   data/The IQ-OTHNCCD lung cancer dataset/

3. Train model:
   python train.py

4. Run app:
   streamlit run app.py
