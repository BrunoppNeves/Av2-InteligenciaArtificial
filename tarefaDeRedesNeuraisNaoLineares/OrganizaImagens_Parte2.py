import cv2
import numpy as np
import matplotlib.pyplot as plt

folderRoot = 'C:\\Users\\bruno\\OneDrive\\Documentos\\GitHub\\Av2-InteligenciaArtificial\\tarefaDeRedesNeuraisNaoLineares\\faces\\faces\\'
individual = ['an2i', 'at33', 'boland', 'bpm', 'ch4f', 'cheyer', 'choon', 'danieln', 'glickman', 'karyadi', 'kawamura',
              'kk49', 'megak', 'mitchell', 'night', 'phoebe', 'saavik', 'steffi', 'sz24', 'tammo']
expressoes = ['_left_angry_open', '_left_angry_sunglasses', '_left_happy_open', '_left_happy_sunglasses', '_left_neutral_open', '_left_neutral_sunglasses', '_left_sad_open', '_left_sad_sunglasses', '_right_angry_open', '_right_angry_sunglasses', '_right_happy_open', '_right_happy_sunglasses', '_right_neutral_open', '_right_neutral_sunglasses', '_right_sad_open', '_right_sad_sunglasses',
              '_straight_angry_open', '_straight_angry_sunglasses', '_straight_happy_open', '_straight_happy_sunglasses', '_straight_neutral_open', '_straight_neutral_sunglasses', '_straight_sad_open', '_straight_sad_sunglasses', '_up_angry_open', '_up_angry_sunglasses', '_up_happy_open', '_up_happy_sunglasses', '_up_neutral_open', '_up_neutral_sunglasses', '_up_sad_open', '_up_sad_sunglasses']
QtdIndividuos = len(individual)
QtdExpressoes = len(expressoes)
Red = 40  # Tamanho do redimensionamento da imagem.
X = np.empty((Red*Red, 0))
Y = np.empty((QtdIndividuos, 0))

for i in range(QtdIndividuos):
    for j in range(QtdExpressoes):
        path = folderRoot+individual[i]+'\\'+individual[i]+expressoes[j]+'.pgm'
        PgmImg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ResizedImg = cv2.resize(PgmImg, (Red, Red))

        VectorNormalized = ResizedImg.flatten('F')
        ROT = -np.ones((QtdIndividuos, 1))
        ROT[i, 0] = 1

        cv2.imshow("Foto", PgmImg)
        cv2.waitKey(0)

        VectorNormalized.shape = (len(VectorNormalized), 1)
        X = np.append(X, VectorNormalized, axis=1)
        Y = np.append(Y, ROT, axis=1)


print(f'Quantidade de amostras do conjunto de dados: {X.shape[1]}')
print('A quantidade de preditores esta relacionada ao redimensionamento!')
print(f'Para esta rodada escolheu-se um redimensionamento de {Red}')
print(f'Portanto, a quantidade de preditores desse conjunto de dados: {
      X.shape[0]}')
print(f'Este conjunto de dados possui {Y.shape[0]} classes')
print('****************************************************************')
print('****************************************************************')
print('***********************RESUMO***********************************')
print('****************************************************************')
print('****************************************************************')
print(f'X tem ordem {X.shape[0]}x{X.shape[1]}')
print(f'Y tem ordem {Y.shape[0]}x{Y.shape[1]}')


# Defina as funções de ativação (g e g_prime) e a estrutura da rede neural
def g(x):
    return 1 / (1 + np.exp(-x))

def g_prime(x):
    return x * (1 - x)

# Defina a estrutura da rede neural (número de camadas, número de neurônios em cada camada, etc.)
num_input_neurons = Red * Red
num_hidden_neurons = 50
num_output_neurons = QtdIndividuos

# Inicialize os pesos da rede neural aleatoriamente
np.random.seed(0)
hidden_weights = np.random.rand(num_hidden_neurons, num_input_neurons + 1)
output_weights = np.random.rand(num_output_neurons, num_hidden_neurons + 1)

# Defina a taxa de aprendizado e o número máximo de épocas
learning_rate = 0.1
max_epochs = 1000

# Defina os dados de treinamento (Xtreino) e rótulos (Ytreino)
Xtreino = X  # Supondo que X seja a entrada de treinamento
Ytreino = Y  # Supondo que Y seja a saída desejada

# Função Forward
def forward(xamostra):
    i_hidden = np.dot(hidden_weights, np.insert(xamostra, 0, 1))
    y_hidden = g(i_hidden)
    i_output = np.dot(output_weights, np.insert(y_hidden, 0, 1))
    y_output = g(i_output)
    return y_hidden, y_output

# Função Backward
def backward(xamostra, d, y_hidden, y_output):
    delta_output = g_prime(y_output) * (d - y_output)
    delta_hidden = g_prime(y_hidden) * np.dot(output_weights[:, 1:].T, delta_output)

    output_weights += learning_rate * np.outer(delta_output, np.insert(y_hidden, 0, 1))
    hidden_weights += learning_rate * np.outer(delta_hidden, np.insert(xamostra, 0, 1))

# Treinamento da rede neural
for epoch in range(max_epochs):
    total_error = 0
    for i in range(Xtreino.shape[1]):  # Assumindo que X é uma matriz de entrada (Red*Red, número de amostras)
        xamostra = Xtreino[:, i]
        d = Ytreino[:, i]

        y_hidden, y_output = forward(xamostra)
        backward(xamostra, d, y_hidden, y_output)
        total_error += np.sum((d - y_output) ** 2)

    total_error /= (2 * Xtreino.shape[1])
    print(f"Época {epoch + 1}/{max_epochs}, Erro Quadrático Médio: {total_error}")

print("Treinamento concluído.")