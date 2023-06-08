import json
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
with open('y_pred.json','r') as f:
    y_pred=json.load(f)
y_pred=[item if item!=3 else 2 for item in y_pred]
test_data =pd.read_json('../Data/test.json')
y_true=[]
for i in range(len(test_data)):
    y_true.append(test_data['label'][i])
f1_micro=f1_score(y_pred,y_true,average='micro')
f1_macro=f1_score(y_pred,y_true,average='macro')
print('f1_micro:',f1_micro)
print('f1_macro:',f1_macro)

cm = confusion_matrix(y_true, y_pred)


labels = ['Supported', 'Refuted', 'Not Enough Information']
report = classification_report(y_true, y_pred)
print(report)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)


thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()