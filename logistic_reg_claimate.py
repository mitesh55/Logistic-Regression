# imoprt libraries :
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report


# load dataset :
data = pd.read_csv(r'E:\ExcelR ass\logistic_reg\dataset\claimants.csv')
# print(data.head())
# print(data.shape)


# drop unneccessery column :
data = data.drop("CASENUM", axis=1)
# print(data.head())
# print(data.isna().sum())
# print(data.info())


# MEAN IMPUTATION :
#for clmsex:
sex_mean = data.CLMSEX.mean()
data["CLMSEX"] = data["CLMSEX"].fillna(sex_mean)

# for clminsur :
insur_mean = data.CLMINSUR.mean()
data["CLMINSUR"] = data["CLMINSUR"].fillna(insur_mean)

#for seatbelt :
belt_mean = data.SEATBELT.mean()
data["SEATBELT"] = data["SEATBELT"].fillna(belt_mean)

#for clmage :
age_mean = data.CLMAGE.mean()
data["CLMAGE"] = data["CLMAGE"].fillna(age_mean)

# loss:
loss_mean = data.LOSS.mean()
data["LOSS"] = data["LOSS"].fillna(loss_mean)
# print(data.info())


# split data into label and features :
y = data.iloc[:,0]
x = data.iloc[:,1:]
# print(x.info())


# split data for training and testing :
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
# print(x_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# prepare model :
model = LogisticRegression()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
# print(predicted)

df = pd.DataFrame(predicted)
df.columns = ["actual"]
df["predicted"] = predicted
# print(df.head())
# plt.hist(data.CLMSEX)
# plt.show()


#for r2 :
r2 = model.score(x_train, y_train)
# print(r2)


# for new dataframe ;
y_pred = model.predict(x)
# print(y_pred.shape)
y_pred_prob = model.predict_proba(x)
# print(y_pred_prob)
y_pred_df = pd.DataFrame({"actual":y, "predicted": y_pred, "prob[0]":y_pred_prob[:,0], "prob[1]":y_pred_prob[:,1]})
# print(y_pred_df.head())


# Crosstab:
# print(pd.crosstab(y_pred_df.actual, y_pred_df.predicted))


# for confusion metrix :

cm = confusion_matrix(y_pred_df.actual, y_pred_df.predicted)
# print(cm)

accuracy = accuracy_score(y_pred_df.actual, y_pred_df.predicted)
# print(accuracy)

# for ruc:
fpr, tpr, threshold = roc_curve(y, y_pred_prob[:,1])
# print(threshold)

df_new = pd.DataFrame({"fpr":fpr, "tpr":tpr, "cutoff":threshold})
# print(df_new.head())
plt.plot(fpr, tpr, color='red')
plt.xlabel('False Positive Rate')
plt.ylabel("True Positive Rate")

plt.plot([0,1],[0,1],'k--')
# plt.show()

#for auc :
auc = roc_auc_score(y, y_pred)
# print(auc)

# classification report:
# print(classification_report(y_pred_df.actual, y_pred_df.predicted))   # at 0.5 :69, 72


# for better cutoff :
prob = model.predict_proba(x)
prob = prob[:,1]
update_df = pd.DataFrame({"actual":y, "pred":0})
update_df.loc[prob>0.55, "pred"]=1                       # at 0.55: 71,71
# print(update_df)

# print(classification_report(update_df.actual, update_df.pred))


# predict new_data :
new_data = pd.DataFrame({"CLMSEX": 1, "CLMINSUR": 0, "SEATBELT": 1, "CLMAGE": 43, "LOSS": 7}, index=[0])
ans_new_data = model.predict(new_data)
# print(ans_new_data)
# plt.hist(data.LOSS)
# plt.show()
