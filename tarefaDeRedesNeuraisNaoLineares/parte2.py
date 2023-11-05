import numpy as np
import matplotlib.pyplot as plt

def Forward(xamostra):
    j = 0
    for W in pesos_da_rede:  # Suponhamos que pesos_da_rede seja uma lista de matrizes de peso
        if j == 0:
            i = np.dot(W, xamostra)
            y = g(i)
        else:
            ybias = np.insert(y, 0, -1)  # Adiciona -1 na primeira posição do vetor y
            i = np.dot(W, ybias)
            y = g(i)
        j += 1


def BackWard(xamostra, d):
    j = len(pesos_da_rede) - 1
    while j >= 0:
        if j + 1 == len(pesos_da_rede):
            delta = g_prime(i[j]) * (d - y[j])
            if j > 0:
                ybias = np.insert(y[j - 1], 0, -1)
                pesos_da_rede[j] += learning_rate * np.outer(delta, ybias)
            else:
                pesos_da_rede[j] += learning_rate * np.outer(delta, xamostra)
        else:
            Wb = np.transpose(pesos_da_rede[j + 1])[:, 1:]  # Remove a coluna que multiplica pelos limiares de ativação
            delta = g_prime(i[j]) * np.dot(Wb, delta)
            ybias = np.insert(y[j - 1], 0, -1)
            pesos_da_rede[j] += learning_rate * np.outer(delta, ybias)
        j -= 1


def calcula_EQM(Xtreino):
    EQM = 0.0
    for amostra in Xtreino:
        xamostra = amostra[0]  # Suponhamos que a amostra seja uma lista onde o primeiro elemento é a entrada
        d = amostra[1]  # O segundo elemento é o rótulo

        EQI = 0.0
        Forward(xamostra)

        for j in range(len(d)):
            EQI += (d[j] - y[-1][j]) ** 2  # y[-1] é a saída da camada de saída

        EQM += EQI

    EQM /= (2 * len(Xtreino))
    return EQM

def MLP(Xtreino, CritérioParada, MaxEpoch):
    EQM = 1.0
    Epoch = 0

    while EQM > CritérioParada and Epoch < MaxEpoch:
        for amostra in Xtreino:
            xamostra = amostra[0]  # Suponhamos que a amostra seja uma lista onde o primeiro elemento é a entrada
            d = amostra[1]  # O segundo elemento é o rótulo

            # Chamando as funções Forward e BackWard (substitua por seus próprios métodos)
            Forward(xamostra)
            BackWard(xamostra, d)

        EQM = calcula_EQM()  # Suponhamos que você tenha uma função para calcular o erro médio quadrático
        Epoch += 1

# Exemplo de uso
Xtreino = [(entrada1, rotulo1), (entrada2, rotulo2), ...]  # Substitua pelas suas amostras de treinamento
CritérioParada = 0.001  # Defina o critério de parada desejado
MaxEpoch = 1000  # Defina o número máximo de épocas

treinamento_rede_neural(Xtreino, CritérioParada, MaxEpoch)
