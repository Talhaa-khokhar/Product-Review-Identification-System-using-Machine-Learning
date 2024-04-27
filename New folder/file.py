#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


# Load data
data = pd.read_csv('fake reviews dataset.csv')


# In[3]:


rows, cols = data.shape
print("The dataset contains",rows,"rows and",cols,"columns" )


# In[37]:


data.head()


# In[5]:


data.info()


# In[39]:


data.describe()


# In[40]:


data.info()


# In[42]:


data.head(20)


# In[43]:


data.tail(17)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


data.size


# In[7]:


data.duplicated().sum()


# In[8]:


data.describe()


# In[9]:


data.nunique()


# In[10]:


data.isnull().sum()


# In[11]:


# Preprocess text
data = data.dropna(subset=['text_', 'label'])
text = data['text_'].str.lower()
text = text.str.replace('[^\w\s]','', regex=False) # remove punctuation
text = text.str.replace('\d+', '', regex=False) # remove digits
text


# In[12]:


# Extract features using Bag of Words approach
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(text)


# In[13]:


X


# In[14]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)


# In[15]:


print(type(X_train), X_train.shape )
print(type(X_test), X_test.shape )

print(type(y_train), y_train.shape )
print(type(y_test), y_test.shape )


# In[16]:


# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
pred_labels = model.predict(X_test)


# In[17]:


# Evaluate the performance of the model
acc = accuracy_score(y_test, pred_labels)
acc = round(acc*100,2)
print("Accuracy: ", acc)


# In[18]:


print("Accuracy: ", acc)

report = classification_report(y_test, pred_labels)
print("Classification Report: \n", report)


# In[19]:


# Instantiate the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)


# In[20]:


# Evaluate the model
print("Accuracy:", clf.score(X_test, y_test))


# In[21]:


# Create a pie chart to show the distribution of labels
labels = ['Fake', 'Real']
sizes = data['label'].value_counts(normalize=True)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal') 
plt.title('Distribution of Labels')
plt.show()


# In[22]:


#OR = Original reviews (presumably human created and authentic); CG = Computer-generated fake reviews.


import seaborn as sns

sns.countplot(x='label', data=data)
plt.title('Distribution of Labels')
plt.show()


# In[23]:


data.head()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

# Add a column for the length of each text
data['length'] = data['text_'].apply(len)

# Create a box plot to show the distribution of text length by label
sns.boxplot(x='label', y='length', data=data)
plt.title('Distribution of Text Length by Label')
plt.xlabel('Label')
plt.ylabel('Length')
plt.show()


# In[25]:


data['length']


# In[26]:


data['length'] = data['text_'].apply(len)
# Create a histogram of text length
sns.histplot(data=data, x='length', kde=True)
plt.title('Distribution of Text Length')
plt.xlabel('Length')
plt.show()


# In[27]:


import re
import string

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove digits and punctuation
    text = re.sub('[^a-z\s]+', '', text)
    
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    
    return text


# In[28]:


import pandas as pd
import numpy as np

# Generate 40432 rows of random purchasing history data
user_ids = np.random.randint(10000, size=40432)
product_ids = np.random.randint(1000, size=40432)
purchasing_history = pd.DataFrame({'user_id': user_ids, 'product_id': product_ids})

# Save purchasing history data to CSV file
purchasing_history.to_csv('purchasing_history.csv', index=False)


# In[ ]:





# In[29]:


# Load purchasing history data from CSV file into pandas DataFrame
df = pd.read_csv('purchasing_history.csv')
df.shape


# In[30]:


df.head()


# In[31]:


# Load fake review data from CSV file into pandas DataFrame
fake_reviews = pd.read_csv('fake reviews dataset.csv')

# Preprocess text data
fake_reviews['text_'] = fake_reviews['text_'].apply(preprocess_text)

# Load purchasing history data from CSV file into pandas DataFrame
purchasing_history = pd.read_csv('purchasing_history.csv')

fake_reviews['user_id'] = purchasing_history['user_id']


# Aggregate purchasing history by user ID
user_purchases = purchasing_history.groupby('user_id').agg({'product_id': 'count'}).reset_index()
user_purchases = user_purchases.rename(columns={'product_id': 'num_purchases'})

# Merge purchasing history data with fake review data
fake_reviews = pd.merge(fake_reviews, user_purchases, on='user_id', how='left')

# Extract feature vector for each review
features = fake_reviews[['category', 'label', 'rating', 'text_', 'num_purchases']]


# In[32]:


features


# In[33]:


data.head()


# In[34]:


def is_review_genuine(review_text, user_id, purchasing_history):
    """
    Check if a review is genuine based on the user's purchasing history.

    Args:
        review_text (str): The text of the review.
        user_id (int): The ID of the user who wrote the review.
        purchasing_history (pandas.DataFrame): A DataFrame containing the user's purchasing history.

    Returns:
        bool: True if the review is genuine, False otherwise.
    """
    # Check if the review text appears in the user's purchasing history
    if review_text.lower() in purchasing_history[purchasing_history['user_id'] == user_id]['product_id'].astype(str).str.lower().tolist():
        return True
    else:
        return False
review_text = "Missing information on how to use it, but it is a great product for the price!  I"
user_id = 1610

if is_review_genuine(review_text, user_id,purchasing_history):
    print("This is a genuine review.")
else:
    print("This is a fake review.")


# In[35]:


import pandas as pd

def is_review_genuine(review_text, user_id, purchasing_history):
    """
    Check if a review is genuine based on the user's purchasing history.

    Args:
        review_text (str): The text of the review.
        user_id (int): The ID of the user who wrote the review.
        purchasing_history (pandas.DataFrame): A DataFrame containing the user's purchasing history.

    Returns:
        bool: True if the review is genuine, False otherwise.
    """
    # Check if the user has any purchasing history
    if user_id not in purchasing_history['user_id'].tolist():
        return False
    else:
        # Check if the review text appears in the user's purchasing history
        if review_text.lower() in purchasing_history[purchasing_history['user_id'] == user_id]['product_id'].astype(str).str.lower().tolist():
            return True
        else:
            # Check if the product_id of the review exists in the user's purchasing history
            def get_product_id(text):
                # Implementation of function to extract product id from review text
                # This can be replaced with actual implementation
                return None
            product_id = get_product_id(review_text)
            if product_id is not None and product_id.lower() in purchasing_history[purchasing_history['user_id'] == user_id]['product_id'].astype(str).str.lower().tolist():
                return True
            else:
                return False

# Sample usage
purchasing_history = pd.DataFrame({'user_id': [4444, 2222, 1111], 'product_id': ['A123', 'B456', 'C789']})

review_text = "I love this product! It's amazing."
user_id = 12345

if is_review_genuine(review_text, user_id, purchasing_history):
    print("This is a genuine review.")
else:
    print("This is a fake review.")
    
review_text = "Love this!  Well made, sturdy, and very comfortable.  I love it!Very pretty!"
user_id = 9195

if is_review_genuine(review_text, user_id, purchasing_history):
    print("This is a genuine review.")
else:
    print("This is a fake review.")
    
review_text = "love it, a great upgrade from the original.  I've had mine for a couple of years"
user_id = 8698

if is_review_genuine(review_text, user_id, purchasing_history):
    print("This is a genuine review.")
else:
    print("This is a fake review.")


# In[36]:


df


# In[ ]:




