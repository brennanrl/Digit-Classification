
# coding: utf-8

# In[2]:


# to support python 2 and 3
from __future__ import division, print_function, unicode_literals

# common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# to plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# where to save figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# In[3]:


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist


# In[4]:


X, y = mnist["data"], mnist["target"]
X.shape


# In[5]:


y.shape


# In[6]:


28*28


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()


# In[8]:


y[24500]


# In[9]:


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# In[10]:


plot_digit(X[23])


# In[11]:


plot_digit(X[233])


# In[12]:


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# In[13]:


plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()


# In[14]:


y[36000]


# In[15]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[16]:


import numpy as np 

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[17]:


#BINARY CLASSIFIER for 5
y_train_5 = (y_train == 5) #finds all 5's in train set
y_test_5 = (y_test == 5) # finds all 5 in test set


# In[18]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)


# In[19]:


sgd_clf.predict([some_digit])


# In[20]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[21]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


# In[22]:


# maybe not that good, lets look at one that just guesses 5


# In[23]:


from sklearn.base import BaseEstimator 
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# In[24]:


from sklearn.model_selection import cross_val_score
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# In[25]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


# In[26]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)


# In[27]:


y_train_perfect_predictions = y_train_5


# In[28]:


confusion_matrix(y_train_5, y_train_perfect_predictions)


# In[29]:


from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)


# In[30]:


4344 / (4344 + 1307)


# In[31]:


recall_score(y_train_5, y_train_pred)


# In[32]:


4344 / (4344 + 1077)


# In[33]:


from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


# In[34]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, 
                             method="decision_function")


# In[35]:


from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[36]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])

plt.show()


# In[37]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")


# In[38]:


from sklearn.metrics import roc_curve 

y_scores_forest = y_probas_forest[:, 1] # score is proba of positive class because that creates a matrix
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# In[39]:


# this is some fookin ai here lads and lassies
forest_clf.fit(X_train, y_train_5)
forest_clf.predict([some_digit])


# In[46]:


some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()


# In[40]:


# now lets move onto multiclass prediction

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# In[41]:


# bascially just shows that 5 had the highest score from all predicts
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores


# In[42]:


# proof that it is the five column
sgd_clf.classes_


# In[43]:


# that was ova, but if you want ovo, can do
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])


# In[44]:


# shows how many seperate predictors it had to train
len(ovo_clf.estimators_)


# In[45]:


# then lets do the same thing with a random forest
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])


# In[59]:


# lets look at the logic
forest_clf.predict_proba([some_digit])


# In[60]:


cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')


# In[66]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[62]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# In[63]:


from sklearn.preprocessing import MinMaxScaler

scalerminmax = MinMaxScaler()
X_trainminm_scaled = scalerminmax.fit_transform(X_train)
cross_val_score(forest_clf, X_trainminm_scaled, y_train, cv=3, scoring="accuracy")


# In[68]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# In[69]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[70]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums


# In[71]:


np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[72]:


# multilabel classification

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# In[73]:


knn_clf.predict([some_digit])


# In[ ]:


# will take long to run is about 0.97709
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
f1_score(y_multilabel, y_train_knn_pred, average="macro")


# In[86]:


# multioutput classification AND A NOISE FILTER BABY
# this adds nois eto some images

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))

X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test


# In[87]:


some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()


# In[88]:


knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])

plot_digit(clean_digit)


# In[78]:


# this makes a really good classifier for this 
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)


# In[79]:


y_knn_pred = knn_clf.predict(X_test)


# In[80]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_knn_pred)


# In[82]:


# move images by a few pixels, add more items to training, better pred
from scipy.ndimage.interpolation import shift

def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

plot_digit(shift_digit(some_digit, 5, 1, new=100))


# In[84]:


X_train_expanded = [X_train]
y_train_expanded = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape


# In[85]:


knn_clf.fit(X_train_expanded, y_train_expanded)


# In[ ]:


y_knn_expanded_pred = knn_clf.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_knn_expanded_pred)


# In[ ]:


import Image 

imageOfBlur = Image.open("")

