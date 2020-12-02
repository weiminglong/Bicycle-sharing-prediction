import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset/london_merged.csv")
data.drop(['timestamp'], inplace=True, axis=1)

bins = [-1, 1000, 2000, 3000, 4000, 100000]
dpy = data['cnt'].to_numpy()
r = pd.cut(dpy, bins)
data_y = r.codes

data.drop(['cnt'], inplace=True, axis=1)
# data = (data - data.min()) / (data.max() - data.min())

x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=0.33, random_state=50)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

model = MLPClassifier(solver='sgd', activation='relu', hidden_layer_sizes=(100, 50), alpha=0.01, batch_size=30,
                      learning_rate='constant')
model.fit(x_train, y_train)
result = model.predict(x_test)
print(accuracy_score(result, y_test))
