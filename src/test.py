from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
X = breast_cancer.data[:, :10]
y = breast_cancer.target

columns = ['半径', 'テクスチャ', '周囲の長さ', '面積', 'なめらかさ', 'コンパクト性', 'へこみ', 'へこみの数', '対称性', 'フラクタル次元']

df = DataFrame(data=X[:, :10], columns=columns)
df['目的変数'] = y
X = df[['面積', 'へこみ']].values
y = df['目的変数'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.preprocessing import StandardScaler
# StandardScalerのインスタンスを作成する
sc = StandardScaler()
# 訓練データの平均と標準偏差を計算する
sc.fit(X_train)

# 訓練データの標準化
X_train_std = sc.transform(X_train)

# テストデータの標準化
# テストデータは訓練データの平均と標準偏差を用いて変換する
X_test_std = sc.transform(X_test)

from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=25, probability=True, random_state=42)

svc.fit(X_train_std, y_train)

# テストデータの予測
pred = svc.predict(X_test_std)
# 予測の確認
# print(pred)

# 確率
# proba = svc.predict_proba(X_test_std)
# print(proba[0])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))