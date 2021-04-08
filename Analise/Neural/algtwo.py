import numpy as np
from random import uniform, randint
from math import exp, pow

def rna_treino(data,cl,TMAX,ns,nx,nz,ny):

    alfa = 0.5

    x = np.zeros(nx+1,dtype=float)
    z = np.zeros(nz+1,dtype=float)
    y = np.zeros(ny,dtype=float)

    v = np.zeros((nx+1,nz), dtype=float)
    w = np.zeros((nz+1,ny), dtype=float)

    se = np.zeros(ny, dtype=int)
    sr = np.zeros(ny, dtype=int)

    for i in range(nx+1):
        for j in range(nz):
            v[i][j] = uniform(-1, 1)

    for i in range(nz+1):
        for j in range(ny):
            w[i][j] = uniform(-1, 1)

    for t in range(TMAX):
        line = randint(0,ns-1)

        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1.

        for k in range(ny):
            se[k] = -1
        se[cl[line]-1] = 1

        for j in range(nz):
            z[j] = 0.
            for i in range(nx+1):
                z[j] += x[i]*v[i][j]
            z[j] = -1+(2/(1+exp(-z[j])))
        z[nz] = 1.

        for k in range(ny):
            y[k] = 0.
            for j in range(nz+1):
                y[k] += z[j]*w[j][k]
            y[k] = -1+2/(1+exp(-y[k]))
            if y[k]>=0.:
                sr[k] = +1
            else:
                sr[k] = -1

        errou = False
        for k in range(ny):
            if se[k]!=sr[k]:
                errou = True
                break

        if(errou):
            dz = np.zeros(nz,dtype=float)
            dy = np.zeros(ny,dtype=float)

            for k in range(ny):
                dy[k] = (se[k] - sr[k])*(1+y[k])*(1-y[k])/2.

            for j in range(nz):
                for k in range(ny):
                    dz[j] += dy[k]*w[j][k]
                dz[j] = dz[j]*(1+z[j])*(1-z[j])/2.

            # ATUALIZA PESOS
            for i in range(nx+1):
                for j in range(nz):
                    v[i][j] += alfa*x[i]*dz[j]

            for j in range(nz+1):
                for k in range(ny):
                    w[j][k] += alfa*z[j]*dy[k]

    return v, w


def normaliza(dados):

    ns = len(dados)
    nx = len(dados[0])-1
    cl = np.zeros(ns, int)
    d = np.zeros((ns, nx), float)
    print(ns, nx)

    for j in range(nx):
        maior = menor = float(dados[0][j])
        for i in range(1, ns):
            a = float(dados[i][j])
            if a > maior:
                maior = float(dados[i][j])
            elif a < menor:
                menor = float(dados[i][j])

        for i in range(ns):
            d[i][j] = 0.001 + 0.998 * (float(dados[i][j])-menor)/(maior-menor)

        for i in range(ns):
            cl[j] = int(dados[i][nx])

    return d, cl



def le_arquivo(arq):
    f = open(arq, "r")

    dados_str = []

    for line in f:
        dados_str.append(line.strip('\n').split(','))
    f.close()

    return dados_str

# PROGRAMA PRINCIPAL
dados = []
cl = []
dados = le_arquivo("iris_mlp.txt")
dados, cl = normaliza(dados)
ns = len(dados)
nx = len(dados[0])
nz = 7
ny = 3
ns = len(dados)
tmax = [50, 100, 500, 1000, 2000, 3000, 5000, 8000]
qt = 20

for i in range(len(tmax)):
    prec = 0
    for j in range(qt):
        v, w = rna_treino(dados, cl, tmax[i], ns, nx, nz, ny)
        prec += rna_teste(dados, cl, v, w, ns, nz, ny)
    print("TMAX = ". tmax[i], "Precisao = ", prec/qt)