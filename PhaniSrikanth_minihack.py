import xgboost as xgb
import numpy as np
import pandas as pd

train = pd.read_csv("train_63qYitG.csv")
test = pd.read_csv("test_XaoFywY.csv")

train['Gender'] = train['Gender'].replace(to_replace = {'Male': 0, 'Female': 1})
test['Gender'] = test['Gender'].replace(to_replace = {'Male': 0, 'Female': 1})

type_of_cab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
train['Type_of_Cab'] = train['Type_of_Cab'].replace(to_replace = type_of_cab)
test['Type_of_Cab'] = test['Type_of_Cab'].replace(to_replace = type_of_cab)


confi_index = {'A': 0, 'B': 1, 'C': 2}
train['Confidence_Life_Style_Index'] = train['Confidence_Life_Style_Index'].replace(to_replace = confi_index)
test['Confidence_Life_Style_Index'] = test['Confidence_Life_Style_Index'].replace(to_replace = confi_index)

train['Surge_Pricing_Type'] = train['Surge_Pricing_Type'] - 1

X_train = train.copy()
X_test = test.copy()

from sklearn.preprocessing import LabelEncoder
print("Label Encoding...")
for f in ['Destination_Type']:
    lbl = LabelEncoder()
    lbl.fit(list(X_train[f].values) + list(X_test[f].values))
    X_train[f] = lbl.transform(list(X_train[f].values))
    X_test[f] = lbl.transform(list(X_test[f].values))

features = np.setdiff1d(train.columns, ['Trip_ID', 'Surge_Pricing_Type'])

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 3,
                "seed": 2016, "tree_method": "exact"}
dtrain = xgb.DMatrix(X_train[features], X_train['Surge_Pricing_Type'], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

submit = pd.DataFrame({'Trip_ID': test['Trip_ID'], 'Surge_Pricing_Type': test_preds+1})
submit.to_csv("XGB.csv", index=False)
