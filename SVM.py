
 import numpy as np
 import pandas as pd
 import matplotlib.pyplot as plt
 import matplotlib.mlab as mlab

 from sklearn.preprocessing import StandardScaler
 from sklearn.metrics import confusion_matrix
 from sklearn.svm import SVC

 import itertools
 import seaborn
 from sklearn.metrics import r2_score

# Confusion Matrix grafigi icin fonksiyon
def plot_confusion_matrix(cm, classes,

title='Confusion matrix', cmap=plt.cm.Blues):
 data
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45) plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product (range(cm.shape[0]), range(cm.shape[1])): plt.text(j, i, format(cm[i, j], fmt),
horizontalalignment="center",
color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
=
pd.read_csv("heart.csv")

data = pd.read_csv("heart.csv")
 data.head()
 data.target.value_counts()
 # target degerleri ve sayıları

 seaborn.countplot(x="target", data=data, palette="bwr")

 class_names = np.array(['0', '1'])


 # VERI ON ISLEME

 #x = data.iloc[:,0:-1]
 x = data.drop(labels="target", axis= 1)
 y= data.iloc[:, -1:].values


 ## OZNITELIK OLCEKLEME ##
 # NORMALIZE ISLEMI
 x_norm = (x = np.min(x)) / (np.max(x) - np.min(x)).values

 #VERILERIN EGITIM-TEST OLARAK BOLUNMESI
 from sklearn.model_selection import train_test_split
 x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size= 0.33, random_state=0)
 # x: bagımsız degisken, y: bagımlı degisken, test verisi oran: test_size, bolunme sekli: random_state

 # Target 'taki 0 ve 1 degerlerinin sayıları
 no_heart_disease y[y[:,0]== 0]
 no_heart_disease = len(no_heart_disease) 66 heart_disease = y[y[:0]==1]
 heart_disease = len(heart disease)
 print("Sınıf 0: "+str(no_heart_disease)+"\nSınıf 1: "+str(heart_disease))


#CONFUSION MATRIX - LINEAR SVM
confusion_matrix(y_test, predicts_in)
print("Confusion Matrix (linear kernel): \n"+str(cm_ln))
plot_confusion_matrix(cm_ln, class_names)

 #print("Accuracy: "+str( (cm_ln[0][0]+cm_Ln[1][1]) / (sum(cm_Ln[0])+sum(cm_Ln[1]))))
 print("Test Accuracy of SVM Algorithm (Linear): {:.2f}%".format(classification_ln.score (x_test,y_test)*100))
 print("TPR rate (Sensitivity-Recall): "+ str(cm_in[1][1])+" / "+str(cm_1n[1][0]+cm_1n[1][1])+" = "+str(cm_1n[1][1]/(cm_1n[1][0]+cm_1n[1][1]})\)