import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# 3. Url'den kırmızı şarap verilerini al
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.head(1000))
# 4. Dataseti train ve test olarak ayır.
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# 5. Data Önişleme
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# 6. Hyperparametreleri belirle
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 7. Cross Validation ile tuning
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

# 8. Tüm verisetini refit'e yolla
clf.refit()

# 9. Modeli test et
pred = clf.predict(X_test)
pred2 = clf.predict(X_train) #train için deneme yap
print("Mean Squared Error of train dataset:",mean_squared_error(y_train,pred2))
print("Our r2_score:",r2_score(y_test, pred))
print("Mean Squared Error of test dataset",mean_squared_error(y_test, pred))
