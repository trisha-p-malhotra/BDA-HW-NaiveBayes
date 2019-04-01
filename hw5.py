"""
__desc__: "Big Data Analytics - homework 5 : Naive Bayes classifier 
           on 20 newsgroup dataset"
__author__: "Trisha P Malhotra"

"""
from __future__ import print_function
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# Loading dataset
"""

# For four given categories: 

sub_categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=sub_categories,shuffle=True, random_state=42)
newsgroups_test= fetch_20newsgroups(subset='test', categories=sub_categories,shuffle=True, random_state=42)

"""

# For all categories
newsgroups_train = fetch_20newsgroups(subset='train', categories=None,shuffle=True, random_state=42)
newsgroups_test= fetch_20newsgroups(subset='test', categories=None,shuffle=True, random_state=42)

print('Data set loaded successfully')

# order of labels in `target_names` can be different from `categories`
target_names = newsgroups_train.target_names
print(target_names)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


newsgroups_train_size = size_mb(newsgroups_train.data)
newsgroups_test_size = size_mb(newsgroups_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(newsgroups_train.data), newsgroups_train_size))
print("%d documents - %0.3fMB (test set)" % (
    len(newsgroups_test.data), newsgroups_test_size))

print()


# split a training set and a test set
y_train, y_test = newsgroups_train.target, newsgroups_test.target

# Training data feature extraction
print("Extracting features from the training data using a sparse vectorizer")
print("Using HashingVectorizer:")

vectorizer = HashingVectorizer(stop_words = 'english', alternate_sign = False, n_features = 2 ** 16)
X_train = vectorizer.transform(newsgroups_train.data)
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Sparsity for training data")
print(X_train.nnz / float(X_train.shape[0]))

print()
print("*********************************")
# Test data feature extraction
print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(newsgroups_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)

print("Sparsity for testing data")
print(X_test.nnz / float(X_test.shape[0]))



def MultiNB(clf):

    print("Training: ")
    print(clf)

    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("Training complete")
    #print("Train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    #print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(y_test, pred)
    print("=======================================")
    print()
    print("accuracy:   %0.3f" % score)
    print()
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []

# Train sparse Naive Bayes classifiers
print("*********************************")
print("Naive Bayes")
results.append(MultiNB(MultinomialNB(alpha=0.1)))
print("*********************************")
#print(results)


"""
"Plotting bargraoh "
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)
plt.title("Results")
plt.barh(indices, score, .1, label="score", color='blue')
plt.barh(indices + .2, training_time, .1, label="training time", color='m')
plt.barh(indices + .4, test_time, .1, label="test time", color='orange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

plt.show()
"""