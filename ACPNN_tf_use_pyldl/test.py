from pyldl.utils import load_dataset
from pyldl.algorithms import ACPNN
from pyldl.metrics import score

from sklearn.model_selection import train_test_split

dataset_name = 'emotion6'
X, y = load_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,test_size=0.2,random_state=42)
model = ACPNN()
model.fit(X_train, y_train,learning_rate=5e-3, epochs=3000)

y_pred = model.predict(X_test)
print(score(y_test, y_pred))