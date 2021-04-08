import numpy as np
from random import uniform, randint

def rna_treino(data,TMAX,nx,ns):
    alfa = 0.4                # CONSTANTE DE TREINAMENTO
    x = np.zeros(nx+1,float)  # NOS DA CAMADA DE ENTRADA
    v = np.zeros(nx+1,float)  # PESOS DE ENTRADA PARA SAIDA
    for i in range(nx+1):     # INICIALIZANDO OS PESOS
        v[i] = uniform(-1, 1) # COM VALORES ENTRE -1 E +1

    for t in range(TMAX):
        line = randint(0,ns-1)
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1
        se = int(data[line][nx])  # SAIDA ESPERADA

        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if(y >= 0):              # SAIDA REAL
            sr = +1
        else:
            sr = -1
        if(se!=sr):
            for i in range(nx+1):
                v[i] += alfa*(se - sr)*x[i]
    return v

def rna_teste(data,v,nx,ns):

    x  = np.zeros(nx+1,float)  # NOS DA CAMADA DE ENTRADA
    qa = 0

    for line in range(ns):
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1
        se = int(data[line][nx])  # SAIDA ESPERADA

        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if(y >= 0):              # SAIDA REAL
            sr = +1
        else:
            sr = -1
        if(se==sr):
            qa += 1

    return qa*100./ns

def le_arquivo(arq):
    f = open(arq,"r")

    dados_str = []

    for line in f:
        dados_str.append(line.strip('\n').split(','))
    f.close()

    return dados_str

# PROGRAMA PRINCIPAL
dados = []
dados = le_arquivo("iris_perceptron.txt")
nx = len(dados[0])-1   # qtd. camada de entrada
ns = len(dados)        # qtd. dados treino
tmax = [50,100,500,1000,2000,3000,5000,8000,10000]
qt = 30

for i in range(len(tmax)):
    prec = 0
    for j in range(qt):
        v = rna_treino(dados,tmax[i],nx,ns)
        prec += rna_teste(dados,v,nx,ns)
    print("TMAX = ",tmax[i], "Precisão = ",prec/qt)
