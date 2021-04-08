import numpy as np
from math import exp
from random import uniform, randint

# dados, classe, camada de entrada, camada oculda, camada de saída
def rna_treino(data,cl,TMAX,nx,ny,nz):

    # INICIALIZANDO AS VARIÁVEIS
    alfa = 0.5
    
    x  = np.zeros(nx+1, dtype=float)
    #: [0. 0. 0. 0. 0.]
    z  = np.zeros(nz+1, dtype=float)
    #: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    v = np.zeros((nx+1,nz), dtype=float)
    '''
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    '''
    w = np.zeros((nz+1), dtype=float)
    #: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    vo = np.zeros((nx+1,nz), dtype=float)
    '''
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    '''
    
    wo = np.zeros((nz+1), dtype=float)
    #: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    # PASSO 0 - INICIALIZANDO OS PESOS
    for i in range(nx+1):
        for j in range(nz):
            v[i][j] = uniform(-1, 1)
    
    #print(v)
    '''
    [[-0.03020829 -0.77628037  0.12333611  0.35808559  0.67453966 -0.60020937 0.70131685 -0.97601867  0.55758595 -0.50956603 -0.67548402]
     [-0.27192754  0.95209154 -0.47502872  0.31966703  0.38305455  0.63101762 0.76340539  0.80785642  0.052973   -0.97844602  0.45416597]
     [ 0.18565777 -0.25920793 -0.35481019 -0.12425173  0.6161889   0.47583647 -0.77032793  0.90933517  0.6358698  -0.13397316  0.11185965]
     [-0.80328932 -0.35315027 -0.23856521  0.08765729  0.10138592 -0.21677146 -0.677488   -0.58433908  0.62079069 -0.35161894 -0.26686365]
     [ 0.86371107  0.480022    0.34212525  0.07446567  0.95181453 -0.20734653 -0.75314368  0.42533066 -0.2738269   0.90057389  0.46404628]]
    '''

    # inicilização de pesos de Nguyen-Widrow
    vn = np.zeros(nz, dtype=float)
    beta = 0.7*pow(nz,1/nx)
    for j in range(nz):
        vn[j] = 0
        for i in range(nx):
            vn[j] = v[i][j]*v[i][j]
        vn[j] = pow(vn[j],1/2)

    for i in range(ny):
        for j in range(nz):
            vo[i][j] = v[i][j] = beta*v[i][j]/vn[j]
        vo[nx][j] = v[nx][j] = beta*uniform(-1, 1)


    for i in range(nz+1):
        wo[i] = w[i] = uniform(-1, 1)

    # PASSO 1 - LOOP DE TREINAMENTO
    for t in range(TMAX):
        # PASSO 2 - SELECIONA REGISTRO
        line = randint(0,len(data)-1)


        # PASSSO 3 - ATRIBUI A CAMADA DE ENTRADA
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1.

        se = cl[line]

        # PASSO 4 - PROPAGA SINAL PARA A CAMADA OCULTA E
        # CALCULA SAIDA DESTA CAMADA
        for j in range(nz):
            z[j] = 0.
            for i in range(nx+1):
                z[j] += x[i]*v[i][j]
            #z[j] = 1/(1+exp(-z[j]))
            z[j] = -1+(2/(1+exp(-z[j])))

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

        if sr!=se:
            # ERRO DA CAMADA DE SAÍDA
            dz = np.zeros(nz)
            #dy = (se - sr)*y*(1-y)
            dy = (se - sr)*(1+y)*(1-y)

            # PROPAGA ERRO PARA A CAMADA OCULTA
            for j in range(nz):
                dz[j] = dy*w[j]
                #dz[j] = dz[j]*z[j]*(1-z[j])
                dz[j] = dz[j]*(1+z[j])*(1-z[j])

            # ATUALIZA PESOS
            for i in range(nx+1):
                for j in range(nz):
                    vo[i][j] = v[i][j]
                    v[i][j] += alfa*x[i]*dz[j]

            for j in range(nz+1):
                wo[j] = w[j]
                w[j] += alfa*z[j]*dy

    return v, w


def rna_teste(data,cl,v,w,nx,ny,nz):

    ns = len(data)
    ac = 0

    x  = np.zeros(nx+1, dtype=float)
    z  = np.zeros(nz+1, dtype=float)

    # PASSO 1 - LOOP DE TESTE
    for line in range(ns):

        # PASSSO 2 - ATRIBUI A CAMADA DE ENTRADA
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1

        se = cl[line]

        # PASSO 3 - PROPAGA SINAL PARA A CAMADA OCULTA E
        # CALCULA SAIDA DESTA CAMADA
        for j in range(nz):
            z[j] = 0
            for i in range(nx+1):
                z[j] += x[i]*v[i][j]
            #z[j] = 1./(1.+exp(-z[j]))
            z[j] = -1+2/(1+exp(-z[j]))
        z[nz] = 1.

        # PASSO 4 - PROPAGA SINAL PARA A CAMADA DE SAÍDA E
        # CALCULA SAIDA DESTA CAMADA
        y = 0
        for j in range(nz+1):
            y += z[j]*w[j]
        #y = 1./(1.+exp(-y))
        y = -1+2/(1+exp(-y))

        # TRANSORMA EM SAÍDA DA REDE
        if y>=0:
            sr = +1
        else:
            sr = -1

        if se==sr:
            ac += 1

    return ac*100./ns


def le_arquivo():
    f = open("iris_rna.txt", "r")

    dados_str = []

    for line in f:
        dados_str.append(line.strip('\n').split(','))
    f.close()

    return dados_str

def normaliza(dados):

    ns = len(dados)
    nx = len(dados[0])-1
    cl = np.zeros(len(dados),int)
    d  = np.zeros((ns,nx),dtype=float)

    for j in range(nx):
        maior = menor = float(dados[0][j])
        for i in range(1,ns):
            if float(dados[i][j])>maior:
                maior = float(dados[i][j])
            else:
                if float(dados[i][j])<menor:
                    menor = float(dados[i][j])

        for i in range (ns):
            d[i][j] = 0.001 + 0.998*(float(dados[i][j])-menor)/(maior-menor)

        for i in range (ns):
            cl[i] = int(dados[i][nx])

    return d,cl


# PROGRAMA PRINCIPAL
dados = []
cl = []
dados = le_arquivo()
dados, cl = normaliza(dados)
# print(dados,cl)

# PARÂMETROS DA MLP
nx = len(dados[0])      # camada de entrada
nz = 11                 # camada oculda
ny = 1                  # camada de saída

for tmax in range(500,4000,500):
    # treino da MLP
    v, w = rna_treino(dados,cl,tmax,nx,ny,nz)

    # teste da MLP
    prec = rna_teste(dados,cl,v,w,nx,ny,nz)

    print("TMAX = ",tmax, "Precisão = ",prec)
