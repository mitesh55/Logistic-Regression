import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r'E:\ExcelR ass\logistic_reg\dataset\bank_prblm\bank-full.csv', delimiter=';')
# print(data.head(25))
# print(data.age.shape)


# convert categorical data using LabelEncoder :
le = LabelEncoder()
data["job"] = le.fit_transform(data["job"])
data["marital"] = le.fit_transform(data["marital"])
data["education"] = le.fit_transform(data["education"])
data["default"] = le.fit_transform(data["default"])
data["housing"] = le.fit_transform(data["housing"])
data["loan"] = le.fit_transform(data["loan"])
data["contact"] = le.fit_transform(data["contact"])
data["month"] = le.fit_transform(data["month"])
data["poutcome"] = le.fit_transform(data["poutcome"])
data["y"] = le.fit_transform(data["y"])
# print(data.info())
# print(data.head())


# split dataset into label and features :
y_label = data.iloc[:,16]
# print(y_label.head())
x_features = data.iloc[:,0:16]


# to check for null value :
# print(data.isna().sum())


# plot pairplot and heatmap :
# sns.pairplot(data)
# sns.heatmap(adta.corr(), annot=True)
# plt.show()


# split data for training and testing and preapare model :
x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, train_size=0.8, random_state=1)
model = LogisticRegression()
model.fit(x_train, y_train)
predicted = model.predict(x_test)


y_pred = model.predict(x_features)
y_pred_prob = model.predict_proba(x_features)
# print(y_pred_prob.shape)
# sns.regplot(x=y_label, y=y_pred)
# plt.show()


# for confusion metrix and accuracy_score :
y_df = pd.DataFrame({"actual":y_label, "prediction":y_pred})
# print(y_df.head(50))
# print(pd.crosstab(y_df.actual, y_df.prediction))
cm = confusion_matrix(y_df.actual, y_df.prediction)
# print(cm)
accuracy = accuracy_score(y_df.actual, y_df.prediction)
# print(accuracy)                               # accuracy = 0.88746



# predict for new_data :
new_data = pd.DataFrame({"age":33 , "job":4 , "marital":1 , "education":3 , "default":0,"balance":231, "housing":1, "loan":1,"contact":2, "day":19,"month":6, "duration":1246, "campaign":15, "pdays":102 , "previous":12 , "poutcome":2}, index=[0])
# print(model.predict(new_data))




# ruc_curve and classification_report :
fpr, tpr, threshold = roc_curve(y_label, model.predict_proba(x_features)[:,1])

df_new = pd.DataFrame({"fpr":fpr, "tpr":tpr, "cutoff":threshold})
# plt.plot(fpr, tpr, color="red")
# plt.xlabel('False Positive Rate')
# plt.ylabel("True Positive Rate")
# plt.plot([0,1],[0,1],'k--')
# plt.show()
auc = roc_auc_score(y_df.actual, y_df.prediction)
# print(auc)                                                  # 0.5803
# print(classification_report(y_df.actual, y_df.prediction))     # 0~0.94, 1~0.27



# for better cutoff :
prob = y_pred_prob[:,1]
update_df = pd.DataFrame({"actual":y_label, "pred":0})
update_df.loc[prob>0.235, "pred"]=1
# print(update_df.head(25))
new_auc = roc_auc_score(update_df.actual, update_df.pred)
# print(new_auc)                                                #  auc = 0.6878
# print(classification_report(update_df.actual, update_df.pred))    # 0~0.92,  1~0.44


# dataframe of coefficient for our convenience :
# print(model.coef_)
# print(model.intercept_)
intercept = model.intercept_
coefs = model.coef_
# print(coefs.shape)
coef_df = pd.DataFrame(np.transpose(coefs), x_features.columns, columns=["coefficient"])
# print(coef_df)
# sns.scatterplot(x=y_label, y=y_pred)
# sns.scatterplot(x=y_label, y=x_features)
# plt.show()



# to practice on visulization we take only two features :
data_2 = pd.DataFrame(data["balance"])
data_2.columns = ["balance"]
data_2["duration"] = np.sort(data["duration"])
data_2["y"] = data["y"]
data_2["balance"] = np.sort(data_2["balance"])
# print(data_2.tail(20))
admited = data_2[data_2["y"] == 1]
not_admited = data_2[data_2["y"] == 0]
# print(admited.head())

# plotting of features at 1 and 0 :
# sns.scatterplot(x=admited.balance, y=admited.duration, color="blue", alpha=0.5)
# sns.scatterplot(x=not_admited.balance, y=not_admited.duration, color="orange", alpha=0.5)
# sns.regplot(x=admited.balance, y=admited.duration)
# sns.regplot(x=not_admited.balance, y=not_admited.duration)
# plt.show()

# split new data to preapare model :
x_new = data_2.iloc[:, 0:2]
y_new = data_2.iloc[:,2]
# print(y_new.head())
x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_new, y_new)
new_model = LogisticRegression()
new_model.fit(x_new_train, y_new_train)
new_pred = new_model.predict(x_new_test)
new_accuracy = accuracy_score(y_new_test, new_pred)
# print(new_accuracy)                                  #  0.888

new_intercept = new_model.intercept_
new_coefs = new_model.coef_
new_coefs_df = pd.DataFrame(np.transpose(new_coefs), x_new.columns, columns=["coefficients"])
# print(new_coefs_df)


# for equation line :
y_eq = new_intercept + x_new.balance*new_coefs_df.coefficients[0] + x_new.duration*new_coefs_df.coefficients[1]

y_value = (-1/new_coefs_df.coefficients[1])*(new_intercept + x_new.balance*new_coefs_df.coefficients[0])
x_value = np.array(np.linspace(min(x_new.balance)-2, max(x_new.duration)+2, len(y_value)))
# print(x_value.shape)
# print(y_value.shape)
# sns.regplot(x=admited.balance, y=admited.duration, color="blue")
# sns.regplot(x=not_admited.balance, y=not_admited.duration, color="orange")
# sns.scatterplot(x=admited.balance, y=admited.duration, color="blue", alpha=0.5)
# sns.scatterplot(x=not_admited.balance, y=not_admited.duration, color="orange", alpha=0.5)
# sns.lineplot(x=x_value, y=y_value)
# sns.scatterplot(x_value, y_value)
# plt.legend()
# plt.show()
# print(y_value)
# print(x_value)