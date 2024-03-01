import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
data = pd.read_csv(r"C:\Users\salwa\OneDrive\Desktop\machine learning projects\logistic regression - cancer prediction\data.csv")

sns.heatmap(data.isnull())
#plt.show()
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

for value in data.diagnosis:
  data['diagnosis'] = data['diagnosis'].replace({"M": 1, "B": 0})

y = data["diagnosis"]
x = data.drop(["diagnosis"] , axis=1)
# normalize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

#split the data
x_train, x_test, y_train, y_test = train_test_split(x_scaled , y, test_size=0.3, random_state=42)

lr = LogisticRegression()
lr.fit(x_train , y_train)
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print (f"accuracy = {accuracy}")
print(classification_report(y_test,y_pred))