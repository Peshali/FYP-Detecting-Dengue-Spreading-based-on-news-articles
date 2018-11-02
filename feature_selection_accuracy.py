
# coding: utf-8

# In[1]:


import pandas as pd
import numpy
# Read csv file
df = pd.read_csv('Test.csv')
X = df['Article_content']
Y = df['type']

X = X.fillna("") # fill empty values

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
le = preprocessing.LabelEncoder()
# To convert Y from Object to str and to fit label encoder and return encoded labels
Y = le.fit_transform(Y.astype(str))

from sklearn . model_selection import train_test_split
# Split data set into train and test subsets
X_train , X_test , Y_train , Y_test = train_test_split (X, Y, test_size =0.333 ,random_state =42)

# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# In[2]:


from sklearn.feature_selection import VarianceThreshold
# removes features whose variance doesn't meet 80 %
sel= VarianceThreshold(threshold=(.8 * (1-.8)))
X_train_counts1 = sel.fit_transform(X_train_counts)

print(X_train_counts1.shape)


# In[3]:
#RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
clf = RandomForestClassifier(n_estimators=70,n_jobs=4,class_weight='balanced',oob_score=True,random_state=101,min_samples_leaf=30).fit(X_train_counts1,Y_train)
# text_clf_LogisticRegression = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',  LogisticRegression())])
#RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30)
# from boruta import BorutaPy
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30)
# boruta_selector = BorutaPy(clf,n_estimators='auto',verbose=2)
# boruta_selector.fit(X_train_counts1,Y_train)

clf.fit(X_train_counts1,Y_train)

X_test_dtm = count_vect.transform(X_test)
X_test_dtm = sel.transform(X_test_dtm)
X_test_dtm.shape


# In[4]:


# Performance of NB Classifier
import numpy as np
predicted = clf.predict(X_test_dtm)
print('Performance of NB Classifier',np.mean(predicted == Y_test))


# In[5]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(Y_test, predicted)


# In[6]:


# 2. Using selectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_counts,Y_train)
lsvc = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(X_train_counts,Y_train)
#model = SelectFromModel(lsvc, prefit = True)
model = SelectFromModel(lsvc,threshold=0.15)
X_new = model.transform(X_train_counts)
print('2. using selectFromModel',X_new.shape)


# In[7]:


X_test_dtm = count_vect.transform(X_test)
X_test_dtm = model.transform(X_test_dtm)
X_test_dtm.shape


# In[9]:


# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(X_new,Y_train)


# In[10]:


import numpy as np
predicted = clf.predict(X_test_dtm)
print('Performance of NB Classifier',np.mean(predicted == Y_test))


# In[11]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#X_train_counts.shape
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train_counts,Y_train)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)

X_new = model.transform(X_train_counts)
print('3. Tree based feature selection',X_new.shape)


# In[12]:


X_test_dtm = count_vect.transform(X_test)
X_test_dtm = model.transform(X_test_dtm)
print(X_test_dtm.shape)
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(X_new,Y_train)
import numpy as np
predicted = clf.predict(X_test_dtm)
print('Performance of NB Classifier',np.mean(predicted == Y_test))


# In[13]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=50)
fit = test.fit(X_train_counts, Y_train)
numpy.set_printoptions(precision=3)

features = fit.transform(X_train_counts)
print('4. Univariate Selection', features.shape)


# In[18]:


from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(features,Y_train)


# In[21]:


clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30)

clf.fit(features,Y_train)

X_test_dtm = count_vect.transform(X_test)
X_test_dtm = fit.transform(X_test_dtm)
X_test_dtm.shape


# In[22]:


# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(features,Y_train)


# In[23]:


import numpy as np
predicted = clf.predict(X_test_dtm)
print('Performance of NB Classifier',np.mean(predicted == Y_test))


# In[24]:


model = ExtraTreesClassifier()
model.fit(X_train_counts, Y_train)
print('6. Feature importance',model.feature_importances_.shape)


# In[33]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=50)
fit = test.fit(X_train_counts, Y_train)
numpy.set_printoptions(precision=3)

features = fit.transform(X_train_counts)
print('4. Univariate Selection', features.shape)

from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,max_features=None,min_samples_leaf=30).fit(features,Y_train)


clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30)

clf.fit(features,Y_train)

X_test_dtm = count_vect.transform(X_test)
X_test_dtm = fit.transform(X_test_dtm)
X_test_dtm.shape

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=-1,random_state=101,min_samples_leaf=30).fit(features,Y_train)
import numpy as np
predicted = clf.predict(X_test_dtm)
print('Performance of NB Classifier',np.mean(predicted == Y_test))


# In[29]:




