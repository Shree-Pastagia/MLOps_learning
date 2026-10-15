from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test,predictions)

print("Model tarining completed!")
print("Accuracy:",accuracy)

joblib.dump(model, "model.joblib")
print("Model saved successfully!")

print("Training workflow executed successfully")