# IRIS-DATASET

## Support Vector Classifier (SVC) for Iris Flower Classification
This project uses Support Vector Classifier (SVC) to classify iris flowers into different species based on their measurements.

## Steps of project

1 .Introduction
2. Data preprocessing
3. Model Training
4. Model Evaluation
5. Results
6. Conclusion


### Introduction
The project demonstrates the use of Support Vector Classifier (SVC) with various kernels (RBF, linear, polynomial, sigmoid) to predict the species of iris flowers based on sepal and petal measurements.

### Data preprocessing
Load and Split Data: Load the Iris dataset and split it into training and testing sets:
```python

from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)
```

### Model training
``` python
# Train SVC Models: Train SVC models with different kernels:
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

RBF Kernel with gamma=1.0
svc_rbf01 = SVC(kernel='rbf', gamma=1.0)
svc_rbf01.fit(X_train, Y_train)
Y_predict_rbf01 = svc_rbf01.predict(X_test)
cm_rbf01 = confusion_matrix(Y_test, Y_predict_rbf01)

RBF Kernel with gamma=10.0
svc_rbf10 = SVC(kernel='rbf', gamma=10.0)
svc_rbf10.fit(X_train, Y_train)
Y_predict_rbf10 = svc_rbf10.predict(X_test)
cm_rbf10 = confusion_matrix(Y_test, Y_predict_rbf10)

Linear Kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(X_train, Y_train)
Y_predict_linear = svc_linear.predict(X_test)
cm_linear = confusion_matrix(Y_test, Y_predict_linear)

Polynomial Kernel
svc_poly = SVC(kernel='poly')
svc_poly.fit(X_train, Y_train)
Y_predict_poly = svc_poly.predict(X_test)
cm_poly = confusion_matrix(Y_test, Y_predict_poly)

Sigmoid Kernel
svc_sig = SVC(kernel='sigmoid')
svc_sig.fit(X_train, Y_train)
Y_predict_sig = svc_sig.predict(X_test)
cm_sig = confusion_matrix(Y_test, Y_predict_sig)
```
### Model Evaluation
Confusion Matrix: Evaluate the performance of each SVC model using confusion matrices:
cm_rbf01: Confusion matrix for RBF Kernel with gamma=1.0
cm_rbf10: Confusion matrix for RBF Kernel with gamma=10.0
cm_linear: Confusion matrix for Linear Kernel
cm_poly: Confusion matrix for Polynomial Kernel
cm_sig: Confusion matrix for Sigmoid Kernel

### Results

![Screenshot 2024-07-18 173557](https://github.com/user-attachments/assets/67eee90a-3a9f-4b18-a9ad-a0870133528c)

![Screenshot 2024-07-18 173656](https://github.com/user-attachments/assets/72892870-c0a7-4460-a86d-13eba106b749)

![Screenshot 2024-07-18 173753](https://github.com/user-attachments/assets/2380667a-3711-4b98-b475-d6b75f832072)

![Screenshot 2024-07-18 173823](https://github.com/user-attachments/assets/371093d7-1571-4b6a-ac4e-57603a292680)

![Screenshot 2024-07-18 173613](https://github.com/user-attachments/assets/a66caea1-f4b6-406e-94bf-19ef085e059b)


# Conclusion
This project showcases the application of Support Vector Classifier (SVC) with various kernels to classify iris flowers based on their measurements. The model performance is evaluated using confusion matrices.



