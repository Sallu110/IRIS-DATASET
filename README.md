# IRIS-DATASET

# Support Vector Classifier (SVC) for Iris Flower Classification
This project uses Support Vector Classifier (SVC) to classify iris flowers into different species based on their measurements.

# steps of project
Introduction
Model Training
Evaluation
Conclusion
License

# Introduction
The project demonstrates the use of Support Vector Classifier (SVC) with various kernels (RBF, linear, polynomial, sigmoid) to predict the species of iris flowers based on sepal and petal measurements.

# Model Training
Load and Split Data: Load the Iris dataset and split it into training and testing sets:

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

Train SVC Models: Train SVC models with different kernels:
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# RBF Kernel with gamma=1.0
svc_rbf01 = SVC(kernel='rbf', gamma=1.0)
svc_rbf01.fit(X_train, Y_train)
Y_predict_rbf01 = svc_rbf01.predict(X_test)
cm_rbf01 = confusion_matrix(Y_test, Y_predict_rbf01)

# RBF Kernel with gamma=10.0
svc_rbf10 = SVC(kernel='rbf', gamma=10.0)
svc_rbf10.fit(X_train, Y_train)
Y_predict_rbf10 = svc_rbf10.predict(X_test)
cm_rbf10 = confusion_matrix(Y_test, Y_predict_rbf10)

# Linear Kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, Y_train)
Y_predict_linear = svc_linear.predict(X_test)
cm_linear = confusion_matrix(Y_test, Y_predict_linear)

# Polynomial Kernel
svc_poly = SVC(kernel='poly')
svc_poly.fit(X_train, Y_train)
Y_predict_poly = svc_poly.predict(X_test)
cm_poly = confusion_matrix(Y_test, Y_predict_poly)

# Sigmoid Kernel
svc_sig = SVC(kernel='sigmoid')
svc_sig.fit(X_train, Y_train)
Y_predict_sig = svc_sig.predict(X_test)
cm_sig = confusion_matrix(Y_test, Y_predict_sig)

# Evaluation
Confusion Matrix: Evaluate the performance of each SVC model using confusion matrices:
cm_rbf01: Confusion matrix for RBF Kernel with gamma=1.0
cm_rbf10: Confusion matrix for RBF Kernel with gamma=10.0
cm_linear: Confusion matrix for Linear Kernel
cm_poly: Confusion matrix for Polynomial Kernel
cm_sig: Confusion matrix for Sigmoid Kernel

# Conclusion
This project showcases the application of Support Vector Classifier (SVC) with various kernels to classify iris flowers based on their measurements. The model performance is evaluated using confusion matrices.



