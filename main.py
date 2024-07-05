import numpy as np
import matplotlib.pyplot as plt
# Generate some toy data
X = np.array([[],[], [],[],[]])
y = np.array([[],[], [],[],[]])
w=1
b=1
for i in range(10000):
    z=X.dot(w) +b
    loss=0.5 * np.mean((y-z) **2)
    dloss=y-z
    dw=X.T.dot(dloss)/len(X)
    db=np.mean(dloss)
    lr=0.1
    w-=(-lr)*dw
    b-=(-lr)*db
    if(i%1000==0):
        print("after ",i,"loss: ",loss )
        print("prediction :",z)
