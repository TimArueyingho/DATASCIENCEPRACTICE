#pip installed pydotplus
#pip installed graphviz

import pydotplus
from sklearn.metrics import classification_report
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#load dataset
X=[[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],
   [166,6],[128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],[171,36],[116,25],[196,25]]

Y = ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman',
     'Man','Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man']

data_feature_names = ['height','length of hair']

#they are already separated into dependent and independent variables
#tts

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)

#instantiate model

model =DecisionTreeClassifier()

#fit
model.fit(X_train,Y_train)
predict_y = model.predict([[133,37]])
print(predict_y)
