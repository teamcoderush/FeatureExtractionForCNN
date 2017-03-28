# Load Libraries
import gensim
import logging
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def review_to_words( text ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(text, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)



# Load dataset
url = "../../data/csv/ABSA16_Restaurants_Train_SB1_v2.csv"  # relative dataset URL
dataset = pd.read_csv(url, encoding='latin1')  # reads dataset with headers
train = dataset.groupby('text', as_index=False)['category'].agg({'categories': (lambda x: list(x))})

# Get the number of reviews based on the dataframe column size
num_reviews = train["text"].size
#

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
#

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print("Cleaning and parsing the training set review sentences...")
for i in range(0, num_reviews):
    if ((i + 1) % 250 == 0):
        print("\tReview sentence %d of %d" % (i + 1, num_reviews))
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append(review_to_words(train["text"][i]))
    #

# Map the dataset to a list
lst = []
for t in clean_train_reviews:
    lst += [t.split()]

# Word2Vec Paramaters
MIN_COUNT = 5   # default is 5
SIZE = 100     # default is 100
WORKERS = 5    # default is 5

# logging configurations
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Train the model
model = gensim.models.Word2Vec(lst, min_count=MIN_COUNT, size=SIZE, workers=WORKERS)

# Test Results
# print(model.wv['restaurant'])
print(model.wv.most_similar(positive=['good', 'taste'], negative=['bad']))