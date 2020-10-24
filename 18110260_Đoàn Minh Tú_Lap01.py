##EX1: Predict the onset of diabetes based on diagnostic measures
#load libraries
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
df = pd.read_csv("diabetes.csv", header=0, names=col_names)

print(df.head())

print(df.info())

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'gedigree']
x = df[feature_cols]
y = df.label

#split dataset into training set and test set
#70% training and 30% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1) 

#Building Decision Tree Model
#Create Decision Tree classifer object
clf = DecisionTreeClassifier()

#Train Decision Tree Classifer
clf = clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

perm = PermutationImportance(clf, random_state=1).fit(x_test, y_test)
eli5.show_weight(perm, feature_names = x_test.columns.tolist())

#model accuracy, how ofenten is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Visualizing Decision Trees

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from Ipython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file = dot_data,
                filled = True, rounded = True,
                special_characters = True, feature_names = feature_cols, class_names = ['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('disabetes.png')
Image(grap.create_png())

#EX3:Mushroom Classification
#read data mushrooms
df = pd.read_csv('mushrooms.csv')
df.head()

y = df['class']
x = df.drop('class', axis=1)
y.head()

print(df.dtypes)

from sklearn.preprocessing import LabelEncoder
    Labelencoder = labelEncoder()
for column in df.columns:
  df[column] = labelencoder.fit_transform(df[column])
  
print(df.dtypes)

print(df.head())

print(df.describe())

df = df.drop(["veil-type"], axis=1)

df_div = pd.melt(df, "class", var_name = "Characteristics")
fig, ax = plt.subplots(figsize =(10,5))
p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split=True, data=df_div, inn
df_no_class = df.drop(["class"], axis = 1)
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns))   