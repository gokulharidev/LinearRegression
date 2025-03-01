import numpy as np

class LinearRegression:
    def __init__(self,w,b,lr,epochs):
        self.w=w
        self.b=b
        self.lr=lr
        self.epochs=epochs
    
    def fit(self,x,y):
        for _ in range(self.epochs):
            ypred=x.dot(self.w)+self.b
            error=ypred-y
            dw=1/len(x)*x.T.dot(error)
            db=1/len(x)*np.sum(error)
            self.w-=self.lr*dw
            self.b-=self.lr*db

    def predict(self,x):
        return np.dot(x,self.w)+self.b
    
