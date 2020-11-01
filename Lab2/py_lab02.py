import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
path = 'D:/xAPI-Edu-Data.csv'
data = pd.read_csv(path)
for i in range (1, 17):
    print(data.iloc[:,i].value_counts())
    print("*"*20)



sns.pairplot(data, hue = 'Class')

# heat map
# the hien moi truong tuong quan cua cac features
# neu moi truong tuong quan cua rieng 2 feature de loc
plt.figure(figsize=(14,14))
sns.heatmap(data.corr(),linewidths = 0.1, cmap = "YlGnBu", annot=True)

#plot label
#kiem tr phan bo cua labels co deu hay khong?
#neu can bang thi su dung truc tiep
P_satis = sns.countplot(x="Class", data = data)
# dua tren bieu do thi M ,L va H phan bo tap trung vao nen khong dung truc tiep dc.

plt.figure(figsize= (20,14))
data.raisedhands.value_counts().sort_index().plot.bar()
#bieu do phan bo khong deu

plt.figure(figsize=(10,10))
Raise_hand = sns.boxplot(x = "Class", y = "raisedhands", data=data)
plt.show()
# loai outliers
#bo cac gia tri (20, 65)

Facegrid = sns.FacetGrid(data, hue = "Class")
Facegrid.map(sns.kdeplot, "raisedhands", shade = True),
Facegrid.set(xlim = (0, data.raisedhands.max()))

# data.groupby\n",
data.groupby(['ParentschoolSatisfaction'])['Class'].value_counts()
#neu con hoc tot truong duoc danh gia la good va nguoc lai

pd.crosstab(data['Class'], data['ParentschoolSatisfaction'])

sns.countplot(x = "ParentschoolSatisfaction", data = data, hue = "Class")
# ve mo ta Parent school satisfaction bang bieu do cot

# pie chart\n",
labels = data.ParentschoolSatisfaction.value_counts()
colors = ["blue", "green"]
explode = [0,0]
sizes = data.ParentschoolSatisfaction.value_counts().values
    
plt.pie(sizes, explode=explode, labels = labels, colors = colors)
plt.show

"""**Exercise 3: Credit Card Fraud Detection**"""

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""**read the dataset and check if there is any missing value**

---
"""

path = 'D:/creditcard.csv'
df = pd.read_csv(path)
# missing values
print("missing values:", df.isnull().values.any())

"""**proceed with the visualizations**"""

#plot how balanced the dataset
# plot normal and fraud
count_classes = pd.value_counts(df['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Distributed Transactions")
plt.xticks(range(2), ['Normal', 'Fraud'])
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
# data rất mất cân bằng. nên chưa thể sữ dụng nó được

# heatmap
sns.heatmap(df.corr(), vmin=-1)
plt.show()
#không có nhiều sự tương quan ngoại trừ v2 and Amount

fig, axs = plt.subplots(6, 5, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_facecolor('xkcd:charcoal')
    ax.set_title(df.columns[i])
    sns.distplot(df.iloc[:, i], ax=ax, fit=norm,
                 color="#DC143C", fit_kws={"color": "#4e8ef5"})
    ax.set_xlabel('')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
plt.show()

classes = df['Class']
df.drop(['Time', 'Class', 'Amount'], axis=1, inplace=True)
cols = df.columns.difference(['Class'])
MMscaller = MinMaxScaler()
df = MMscaller.fit_transform(df)
df = pd.DataFrame(data=df, columns=cols)
df = pd.concat([df, classes], axis=1)

import pandas as pd
import numpy as np
def train_validation_splits(df):
    # Fraud Transactions
    fraud = df[df['Class'] == 1]
    # Normal Transactions
    normal = df[df['Class'] == 0]
    print('normal:', normal.shape[0])
    print('fraud:', fraud.shape[0])
    normal_test_start = int(normal.shape[0] * .2)
    fraud_test_start = int(fraud.shape[0] * .5)
    normal_train_start = normal_test_start * 2
    val_normal = normal[:normal_test_start]
    val_fraud = fraud[:fraud_test_start]
    validation_set = pd.concat([val_normal, val_fraud], axis=0)
    test_normal = normal[normal_test_start:normal_train_start]
    test_fraud = fraud[fraud_test_start:fraud.shape[0]]
    test_set = pd.concat([test_normal, test_fraud], axis=0)
    Xval = validation_set.iloc[:, :-1]
    Yval = validation_set.iloc[:, -1]
    Xtest = test_set.iloc[:, :-1]
    Ytest = test_set.iloc[:, -1]
    train_set = normal[normal_train_start:normal.shape[0]]
    Xtrain = train_set.iloc[:, :-1]
    return Xtrain.to_numpy(), Xtest.to_numpy(), Xval.to_numpy(), Ytest.to_numpy(), Yval.to_numpy()

def estimate_gaussian_params(X):
    """
    Calculates the mean and the covariance for each feature.
    Arguments:
    X: dataset
    """
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    return mu, sigma

(Xtrain, Xtest, Xval, Ytest, Yval) = train_validation_splits(df)
(mu, sigma) = estimate_gaussian_params(Xtrain)
# calculate gaussian pdf
p = multivariate_normal.pdf(Xtrain, mu, sigma)
pval = multivariate_normal.pdf(Xval, mu, sigma)
ptest = multivariate_normal.pdf(Xtest, mu, sigma)

def metrics(y, predictions):
    fp = np.sum(np.all([predictions == 1, y == 0], axis=0))
    tp = np.sum(np.all([predictions == 1, y == 1], axis=0))
    fn = np.sum(np.all([predictions == 0, y == 1], axis=0))
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    F1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0
    return precision, recall, F1
def selectThreshold(yval, pval):
    e_values = pval
    bestF1 = 0
    bestEpsilon = 0
    for epsilon in e_values:
        predictions = pval < epsilon
        (precision, recall, F1) = metrics(yval, predictions)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

(epsilon, F1) = selectThreshold(Yval, pval)
print("Best epsilon found:", epsilon)
print("Best F1 on cross validation set:", F1)
(test_precision, test_recall, test_F1) = metrics(Ytest, ptest < epsilon)
print("Outliers found:", np.sum(ptest < epsilon))
print("Test set Precision:", test_precision)
print("Test set Recall:", test_recall)
print("Test set F1 score:", test_F1)