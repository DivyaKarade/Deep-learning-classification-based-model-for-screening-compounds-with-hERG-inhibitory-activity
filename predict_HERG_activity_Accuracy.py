import pandas as pd
from keras.layers import *
from keras.models import Sequential
from numpy.random import seed
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score
from tensorflow import set_random_seed

np.random.seed(100)
set_random_seed(1)
seed(0)

# Load training data set
training_data_df = pd.read_csv("herg_train_activity.csv")

X = training_data_df.drop('Activity_value', axis=1).values
Y = training_data_df[['Activity_value']].values

# Load the separate test data set
test_data_df = pd.read_csv("herg_test_activity.csv")

X_test = test_data_df.drop('Activity_value', axis=1).values
Y_test = test_data_df[['Activity_value']].values

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler1 = MinMaxScaler(feature_range=(0, 1)).fit(X_test)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# summarize the shape of the training and test dataset
print(X.shape, Y.shape)
print(X_test.shape, Y_test.shape)

# identify outliers in the training dataset
pca = PCA(n_components=2)
iso = IsolationForest(contamination=0.1, behaviour='new', n_estimators=100, random_state=0, verbose=0)
pca1 = pca.fit_transform(X)
yhat = iso.fit_predict(pca1)
pca2 = pca.fit_transform(X_test)
yhat_1 = iso.fit_predict(pca2)

# select all rows that are not outliers
mask = yhat != -1
X, Y = X[mask, :], Y[mask]

mask = yhat_1 != -1
X_test, Y_test = X_test[mask, :], Y_test[mask]

# summarize the shape of the updated training & test dataset
print(X.shape, Y.shape)
print(X_test.shape, Y_test.shape)

# Define the model
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(200, input_dim=8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=None,
                kernel_constraint='MaxNorm'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    X,
    Y,
    epochs=200,
    batch_size=100,
    shuffle=True,
    verbose=0,
    validation_data=(X_test, Y_test)
)

# evaluate the model
_, train_acc = model.evaluate(X, Y, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.2f, Test: %.2f' % (train_acc * 100, test_acc * 100))

# Load the data we make to use to make a prediction
Xnew = pd.read_csv("cas.csv").values
Xnew = scaler.transform(Xnew)

# For training set
y_pred = model.predict(X)
y_pred_class = model.predict_classes(X)

# For test set
y_pred_1 = model.predict(X_test)
y_pred_class_1 = model.predict_classes(X_test)

# Make a prediction with the neural network
prediction = model.predict(Xnew)

# Define the cutoff
cutoff = 0.5

# Compute class prediction: y_prediction for train set
y_prediction_1 = np.where(y_pred > cutoff, 1, 0)
y_prediction_2 = np.where(y_pred_class > cutoff, 1, 0)

# Compute class prediction: y_prediction for test set
y_prediction_3 = np.where(y_pred_1 > cutoff, 1, 0)
y_prediction_4 = np.where(y_pred_class_1 > cutoff, 1, 0)

y_prediction = np.where(prediction > cutoff, 1, 0)
print("Herg_Activity {}".format(y_prediction))

# to check the accuracy of prediction of the model
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), prediction[i], Y[i]))
    if (prediction[i]) == 0:
        print("Inactive")
    elif (prediction[i]) == 1:
        print("Active")

# Classification report
print(metrics.classification_report(Y, y_prediction_1, digits=2))

# reduce to 1d array
y_prediction_1 = y_prediction_1[:, 0]
y_prediction_2 = y_prediction_2[:, 0]

# For training set
# ROC AUC
auc = roc_auc_score(Y, y_prediction_1)
print('ROC AUC for training set: %f' % auc)
# classification accuracy: (tp + tn) / (TP + TN + FP + FN)
print('Classification accuracy for training set: %f' % metrics.accuracy_score(Y, y_prediction_2))
# precision tp / (tp + fp)
precision = precision_score(Y, y_prediction_2)
print('Precision for training set: %f' % precision)
# recall: tp / (tp + fn) [Sensitivity/Recall]
recall = recall_score(Y, y_prediction_2)
print('Sensitivity/Recall for training set: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y, y_prediction_2)
print('F1 score for training set: %f' % f1)
# confusion matrix
matrix = confusion_matrix(Y, y_prediction_2)
print(matrix)

# For test set
# reduce to 1d array for train set
y_prediction_3 = y_prediction_3[:, 0]
y_prediction_4 = y_prediction_4[:, 0]

# ROC AUC
auc = roc_auc_score(Y_test, y_prediction_4)
print('ROC AUC for test set: %f' % auc)
# classification accuracy: (tp + tn) / (TP + TN + FP + FN)
print('Classification accuracy for test set: %f' % metrics.accuracy_score(Y_test, y_prediction_4))
# precision tp / (tp + fp)
precision = precision_score(Y_test, y_prediction_4)
print('Precision for test set: %f' % precision)
# recall: tp / (tp + fn) [Sensitivity/Recall]
recall1 = recall_score(Y_test, y_prediction_4)
print('Sensitivity/Recall for test set: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, y_prediction_4)
print('F1 score for test set: %f' % f1)
# confusion matrix for test set
matrix = confusion_matrix(Y_test, y_prediction_4)
print(matrix)

prediction = pd.DataFrame(y_prediction, columns=['prediction']).to_csv('CAS_full_herg.csv')
