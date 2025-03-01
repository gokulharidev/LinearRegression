from LinearRegressionForOnevar.algo import LinearRegression
import numpy as np

x=np.array([[1],[2],[3],[4]])
y=np.array([[33],[44],[55],[66]])
w=1
b=1
lr=0.1
epochs=2000
model=LinearRegression(w,b,lr,epochs)
model.fit(x,y)
x=np.array([[5]])
print(model.predict(x))