import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Scaler kirimata
from sklearn.model_selection import train_test_split # treing testing wenkara genima
from sklearn.neighbors import KNeighborsClassifier # KNN algo eka
from sklearn.metrics import accuracy_score # model eke Accuracy eka belima
from sklearn.metrics import confusion_matrix # accuracy eka matrix akarayen

data = pd.read_csv('Iris.csv')
print(data['Species'].value_counts()) # mehidi karanne "Species kiyana colom eke thibena deta bedila thiyana vidiya saha bedichcha pramanaya
print(data.info) # data set eka gena thorathuru dei, NULL value ema balaganna puluwan
print(data.describe()) # data set eka gena visthara denaganna puluwan

#####################################################################3
# X and Y kadaganna vidiya

# X wenkara genima
X_data = data.iloc[:, 1:5]#mehi kiyawenne [ siyaluma rows , 0 weni colom eke sita 5 wana colom eka thek]

# Y wenkara genima
Y_data = data.iloc[:, -1] # meken kiyanne [ okkoma rows tika, awasana colom eka]

######################################################################

# feacher scaler kirima
scaler = StandardScaler()# object eka heduwa

# X data tika scaler kirima
X = scaler.fit_transform(X_data)

######################################################################
# Treaning saha Testing Data wen kirima
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_data, Y_data, test_size=0.2)# object cll


########################################################################
# K-N-N model eka sedima saha yoda genima
model = KNeighborsClassifier(n_neighbors=1)# n_neighbors kiyanne api Algo eke qwa K agaya
model.fit(X_Train, Y_Train)

###########################################################################
# prediction kirima
pred = model.predict(X_Test)
print(pred[0:5])

#############################################################################
#Accuracy
accuracy = accuracy_score(Y_Test,pred) # accuracy eka balanna denne (aththana Y testing value tika, predict wechcha y value tika)
print(accuracy)

cm = confusion_matrix(Y_Test, pred) # metrix akarayen "mehi vikarnayen nirupanaya karanne prdict karala harigiyapuwa, vikarnayata pitin thiyenne weradichcha"
print(cm)

############################################################################
# Hariyana K value ekak soya genima

correct_sum = [] #ek ek k agayanta harigiya values ganana
for i in range(1,20):
    models = KNeighborsClassifier(n_neighbors=i) # eka eka k agayan deela balanawa
    models.fit(X_Train,Y_Train)
    preds = models.predict(X_Test)
    correct = np.sum(preds == Y_Test) # prediction karapuwa harinam ekathu karanna hari gana
    correct_sum.append(correct) # ekathu karala apu age Array ekata danna
print(correct_sum)

# array eka data frame ekaka vidiyata editipath kirima
result = pd.DataFrame(data=correct_sum)
result.index = result.index+1
print(result.T)









































