"""
This is the implementation of NLP (Natural Language Processing) algorithm for text preprocessing.
"""
# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review = review.lower()
review = review.split()
review = [word for word in review if word not in set(stopwords.words('english'))]