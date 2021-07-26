import numpy as np 
import matplotlib.pyplot as plt 

class Perceptron:
    def __init__(self,n_input,learning_rate): 
        self.w = -1 + 2 * np.random.rand(n_input) #Vector w, peso sinaptico
        self.b = -1 + 2 * np.random.rand() #sesgo
        self.eta = learning_rate
    
    def fit(self, X, Y, epochs=500):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1))
                self.w += self.eta * (Y[i]-y_est) *  X[:,i]
                self.b += self.eta * (Y[i]-y_est)
    
    def predict(self, X): 
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):
            
            y_est[i] = np.dot(self.w, X[:,i]) + self.b #ecuacion de la predicción.
            if y_est[i] >= 0:
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est

def draw_2d(model):
    w1,w2,b = model.w[0],model.w[1],model.b
    li, ls = -2, 2
    plt.plot([li,ls], [(1/w2)*(-w1*(li)-b), (1/w2)*(-w1*(ls)-b)], '--k' )
    plt.show()


def main():       
    i=0
    samp=50
    h = 1.2 + (2.1 -1.2) *  np.random.rand(samp)#altura
    ps = 40 + (180 - 40) *  np.random.rand(samp)#peso
    Y = np.zeros(50)#valor deseado
    ytemp =0
    
    while i < samp:   
        imc = (ps[i]) / ((h[i])**2)
        if imc > 25:
            ytemp = 1
        else: 
            ytemp = 0
        Y[i] = ytemp
        i += 1    

    #Normalizacion de datos
    h = (h - min(h)) / (max(h)- min(h))    
    ps = (ps - min(ps)) / (max(ps)- min(ps))
    #Y = (Y - min(Y)) / (max(Y)- min(Y))

    hp =  np.append(h,ps)
    X = np.reshape(hp, (2,samp))

    neuron = Perceptron(2,0.1)
    neuron.fit(X,Y)
  
    # Dibujo
    _, p = X.shape
    for i in range(p):
    # print(i, X[0,i],X[1,i])
        if Y[i] == 1:

            plt.plot(X[0,i],X[1,i], 'or')
        else:
            plt.plot(X[0,i], X[1,i], 'oy')
        
    
    plt.title('IMC')
    plt.grid('on')
    plt.xlim([-.1,1.1])
    plt.ylim([-.1,1.1])
    plt.xlabel("Altura(m)")
    plt.ylabel("Peso(kg)")    
    draw_2d(neuron)
 
if __name__ == "__main__":
    main()