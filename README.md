Book Review Categorization 


This project classifies Amazon book reviews into 16 categories using four different NLP methods: DistilBERT, SVM, CNN, and Mistral (LLM). The objective is to compare the performance of classical machine learning, deep learning, and prompt-based large language models on a real-world multi-class classification problem. 

The Directory Structure is as follows:

.
├── Data Preparation/                   # Raw data and preprocessing
│   ├── books_data.csv                  # Original Kaggle dataset (book metadata)
│   ├── Books_rating.csv                # Original Kaggle dataset (review text, ratings)
│   └── NLP_Project_Data_Prep.ipynb     # Cleans and merges data, creates stratified splits
├── Testing and Training Data/          # Preprocessed data used in all models
│   ├── 14400_strat_samp_training.csv   # Stratified training set
│   └── 1600_strat_samp_test.csv        # Stratified test set
├── Methods/                            # Model implementations
│   ├── BERT_NLP_Project.ipynb          # DistilBERT fine-tuned model
│   ├── CNN_NLP_Project.ipynb           # Convolutional Neural Network
│   ├── SVM_NLP_Project.ipynb           # Support Vector Machine
│   └── LLM_NLP_Project.ipynb           # Mistral LLM (prompt-based)
├── requirements.txt                    # Project dependencies
└── README.md

Where to Look

Code: All four methods are implemented as separate Jupyter notebooks in the Methods/folder.
Data Files: The preprocessed training and testing CSVs are located in Testing and Training Data.

Testing: https://drive.google.com/uc?export=download&id=1adXdX47B7jiikftExs-Q1GzHnU461JpX
Training: 
https://drive.google.com/uc?export=download&id=1tSYiGAMF2Tcl3EogDEWdMn3-ftvqA4ZN 

Raw Data and Preprocessing: The original Kaggle files and the preprocessing notebook are in the Data Preparation/ folder.

** UPDATE ** Since both files are 2GB+, GitHub isn’t allowing for inclusion of the datasets. Instead they can be accessed in the link below:
 
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews?select=books_data.csv 



Order of Execution: 

1. (Optional) Data Preparation/NLP_Project_Data_Prep.ipynb
- If you want to understand or recreate the stratified training and testing dataset, start by running this notebook. 
- This notebook cleans and merges the original Kaggle datasets into stratified samples used for classification

2. BERT_NLP_Project.ipynb
- This notebook uses DistilBERT’s pretrained model and fine tunes the model to predict and classify book genres based on the fed training data 
- Primarily leverages same approach as that of Lab 7
- Hugging Face transformer

3. CNN_NLP_Project.ipynb
- This notebook uses a Convolutional Neural Network, which is an example of a deep learning neural network, to predict and classify book genres based on the given training data 

4. SVM_NLP_Project.ipynb
This notebook uses a Support Vector Machine Linear Model to predict and classify book genres based on the training data

5. LLM_NLP_Project.ipynb
This notebook uses Mistral AI’s Large Language Model to predict and classify book genres based on the training data

Links to non-standard libraries and example notebooks/tutorials: 
1. Mistral AI: https://console.mistral.ai/home 
2. Pandas: https://pandas.pydata.org/ 
3. SciKit Learn: https://scikit-learn.org/stable/ 
4. TensorFlow: https://www.tensorflow.org/ 
5. Numpy: https://numpy.org/ 
6. Hugging Face: distilbert, transformer https://huggingface.co/docs/transformers/en/model_doc/distilbert 
7. Gensim https://pypi.org/project/gensim/
