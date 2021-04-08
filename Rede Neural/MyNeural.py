import numpy as np
from math import exp
from random import uniform, randint
import csv

def le_arquivo(path):
    
    dataset = []
    fistline = True
    with open(path) as _file:
        data = csv.reader(_file, delimiter=',')
        for line in data:
            
            if not (fistline):
                # Converte os valores da linha para float
                line = [float(elemento) for elemento in line]
                dataset.append(line)
            else:
                fistline = False
    return dataset

# cl : classe, d : dados
def normaliza(dataset, dataset_teste):
    cl = []
    d = []
    cl_t = []
    d_t = []
    
    for data in dataset:
        
        dado = int(data[30])
        if dado == 0:
            dado = -1
            
        cl.append(dado)
        d.append([data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15],data[16],data[17],data[18],data[19],data[20],data[21],data[22],data[23],data[24],data[25],data[26],data[27],data[28]])

    for data in dataset_teste:
        
        dado = int(data[30])
        if dado == 0:
            dado = -1
        
        cl_t.append(dado)
        d_t.append([data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14],data[15],data[16],data[17],data[18],data[19],data[20],data[21],data[22],data[23],data[24],data[25],data[26],data[27],data[28]])

    # normalizacao pela media e desvio padrao
    for j in range(len(d[0])):
        aux = []
        for i in range(len(d)):
            aux.append(d[i][j])
        for i in range(len(d_t)):
            aux.append(d_t[i][j])
            
        media = np.mean(aux)
        desvio = np.std(aux)
        
        for i in range(len(d)):
            d[i][j] = (d[i][j] - media) / desvio
        for i in range(len(d_t)):
            d_t[i][j] = (d_t[i][j] - media) / desvio
        
    return d, cl, d_t, cl_t

# dados, classe, treinamento,  camada de entrada, camada de saída
def rna_treino(data,cl,TMAX,nx ,nz):

    # INICIALIZANDO AS VARIÁVEIS
    alfa = 0.5

    # camada de entrada
    x  = np.zeros(nx+1, dtype=float)
    # camada de oculta
    z  = np.zeros(nz+1, dtype=float)

    #pesos conecta entrada e oculta
    v = np.zeros((nx+1,nz), dtype=float)
    
    # conecta oculta e saida
    w = np.zeros((nz+1), dtype=float)
    
    """
    vo = np.zeros((nx+1,nz), dtype=float)
    wo = np.zeros((nz+1), dtype=float)
    """

    # PASSO 0 - INICIALIZANDO OS PESOS
    for i in range(nx+1):
        for j in range(nz):
            v[i][j] = uniform(-1, 1)

    """
    # inicilização de pesos de Nguyen-Widrow
    vn = np.zeros(nz, dtype=float)
    beta = 0.7*pow(nz,1/nx)
    for j in range(nz):
        vn[j] = 0
        for i in range(nx):
            vn[j] = v[i][j]*v[i][j]
        vn[j] = pow(vn[j],1/2)
    

    for i in range(nx+1):
        for j in range(nz):
            vo[i][j] = v[i][j] = beta*v[i][j]/vn[j]
        vo[nx][j] = v[nx][j] = beta*uniform(-1, 1)
    """

    ## gera pesos da camada da camada oculta 
    for i in range(nz+1):
        w[i] = uniform(-1, 1)

    # PASSO 1 - LOOP DE TREINAMENTO
    for t in range(TMAX):
        # PASSO 2 - SELECIONA REGISTRO
        line = randint(0,len(data)-1)


        # PASSSO 3 - ATRIBUI A CAMADA DE ENTRADA
        for i in range(nx):
            x[i] = data[line][i]
        
        # bias da camada de entrada
        x[nx] = 1.

        # saida esperada da rede
        se = cl[line]

        # PASSO 4 - PROPAGA SINAL PARA A CAMADA OCULTA E
        # CALCULA SAIDA DESTA CAMADA
        for j in range(nz):
            z[j] = 0.
            for i in range(nx+1):
                # somatorio padrao
                z[j] += x[i]*v[i][j]
            #z[j] = 1/(1+exp(-z[j]))
            # funcao de ativacao
            z[j] = -1+(2/(1+exp(-z[j])))

        # bias da camada culta
        z[nz] = 1.

        # PASSO 5 - PROPAGA SINAL PARA A CAMADA DE SAÍDA E
        # CALCULA SAIDA DESTA CAMADA
        y = 0.
        for j in range(nz+1):
            y += z[j]*w[j]
        #y = 1/(1+exp(-y))
        y = -1+2/(1+exp(-y))

        # TRANSORMA EM SAÍDA DA REDE
        if y>=0:
            sr = +1
        else:
            sr = -1

        # verifica se errou
        # back propagation
        if sr!=se:
            # ERRO DA CAMADA DE SAÍDA pag 293
           
            #dy = (se - sr)*y*(1-y)
            # derivada
            dy = (se - sr)*(1+y)*(1-y)/2

            # PROPAGA ERRO PARA A CAMADA OCULTA           
            dz = np.zeros(nz)
            for j in range(nz):
                dz[j] = dy*w[j]
                #dz[j] = dz[j]*z[j]*(1-z[j])
                dz[j] = dz[j]*(1+z[j])*(1-z[j])/2

            # ATUALIZA PESOS
            for i in range(nx+1):
                for j in range(nz):
                    #vo[i][j] = v[i][j]
                    v[i][j] += alfa*x[i]*dz[j]

            for j in range(nz+1):
                #wo[j] = w[j]
                w[j] += alfa*z[j]*dy

    return v, w

# dados, classe, numero de nos entrada,  numero de nos oculta, quantidade de registros, peso, peso
def rna_teste(data, cl, nx, nz, ns, v, w):

    # total de acertos
    ac = 0

    # camada de entrada
    x  = np.zeros(nx+1, dtype=float)
    # camada de oculta
    z  = np.zeros(nz+1, dtype=float)
    
    # PASSO 1 - LOOP DE TREINAMENTO
    for line in range(ns):
     
        # PASSSO 3 - ATRIBUI A CAMADA DE ENTRADA
        for i in range(nx):
            x[i] = data[line][i]
        
        # bias da camada de entrada
        x[nx] = 1.

        # saida esperada da rede
        se = cl[line]

        # PASSO 4 - PROPAGA SINAL PARA A CAMADA OCULTA E
        # CALCULA SAIDA DESTA CAMADA
        for j in range(nz):
            z[j] = 0.
            for i in range(nx+1):
                # somatorio padrao
                z[j] += x[i]*v[i][j]
            #z[j] = 1/(1+exp(-z[j]))
            # funcao de ativacao
            z[j] = -1+(2/(1+exp(-z[j])))

        # bias da camada culta
        z[nz] = 1.

        # PASSO 5 - PROPAGA SINAL PARA A CAMADA DE SAÍDA E
        # CALCULA SAIDA DESTA CAMADA
        y = 0.
        for j in range(nz+1):
            y += z[j]*w[j]
        #y = 1/(1+exp(-y))
        y = -1+2/(1+exp(-y))

        # TRANSORMA EM SAÍDA DA REDE
        if y>=0:
            sr = +1
        else:
            sr = -1

        # verifica se errou
        # back propagation
        if sr==se:         
            ac = ac + 1
           

    return (100 * ac)/ns

data = le_arquivo("treino.csv")
data_teste = le_arquivo("teste.csv")
dados, classe, dados_teste, classe_teste = normaliza(data, data_teste)


# PARÂMETROS DA MLP
nx = len(dados[0])      # camada de entrada
nz = 7                 # camada oculda
TMax = 20000

# treino da MLP
v, w = rna_treino(dados, classe, TMax, nx, nz)

# teste da MLP
prec = rna_teste(dados_teste, classe_teste, nx, nz, len(dados_teste), v, w)
print("TMAX = ", TMax, "Precisão = ", prec)
