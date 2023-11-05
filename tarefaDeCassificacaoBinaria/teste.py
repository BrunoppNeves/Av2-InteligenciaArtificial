import numpy as np
import matplotlib.pyplot as plt

# Carregando dados
dados = np.loadtxt('tarefaDeCassificacaoBinaria\DataAV2.csv', delimiter=',')
X = dados[:, 0:2]
Y = dados[:, 2]
N, p = X.shape

plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.title("Gráfico de Dispersão")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

X = np.concatenate((-np.ones((N, 1)), X), axis=1)

def dividir_dados():
    indexOfOitentaPorCento = int(N * 0.8)

    # Embaralhe X e Y juntos
    embaralhado = np.random.permutation(np.column_stack((X, Y)))

    X_embaralhado = embaralhado[:, :-1]
    Y_embaralhado = embaralhado[:, -1]

    X_treino = X_embaralhado[0:indexOfOitentaPorCento, :]
    Y_treino = Y_embaralhado[0:indexOfOitentaPorCento]
    X_teste = X_embaralhado[indexOfOitentaPorCento: N, :]
    Y_teste = Y_embaralhado[indexOfOitentaPorCento: N]

    return X_treino, Y_treino, X_teste, Y_teste

def sign(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, Y, max_iterations, learning_rate):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for _ in range(max_iterations):
        error_exists = False
        for i in range(num_samples):
            input_sample = X[i]
            target = Y[i]
            u_t = np.dot(weights, input_sample)
            y_t = sign(u_t)

            if target != y_t:
                error_exists = True
                weights = weights + learning_rate * (target - y_t) * input_sample
        if not error_exists:
            break
    return weights

def mean_squared_error(d, u):
    return np.mean((d - u) ** 2)

def train_adaline(X, d, eta, max_epochs, epsilon):
    num_samples, num_features = X.shape
    w = np.random.rand(num_features)

    for epoch in range(max_epochs):
        EQM_anterior = mean_squared_error(d, np.dot(X, w))

        for i in range(num_samples):
            u_t = np.dot(X[i], w)
            error = d[i] - u_t
            w += eta * error * X[i]

        EQM_atual = mean_squared_error(d, np.dot(X, w))

        if abs(EQM_atual - EQM_anterior) <= epsilon:
            break

    return w

def perceptron_test(w, x_unknown):
    u = np.dot(w, x_unknown)
    y_t = sign(u)
    # if y_t == -1:
    #     print("PERCEPTRON: A amostra pertence à classe A")
    # else:
    #     print("PERCEPTRON: A amostra pertence à classe B")
    return y_t

def adaline_test(w, x_unknown):
    u = np.dot(w, x_unknown)
    y_t = sign(u)
    # if y_t == -1:
    #     print("ADALINE: A amostra pertence à classe A")
    # else:
    #     print("ADALINE: A amostra pertence à classe B")
    return y_t

X_treino, Y_treino, X_teste, Y_teste = dividir_dados()

# Funções para métricas de avaliação
def acuracia(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def sensibilidade(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == -1))
    if TP + FN == 0:
        return 0  # Evita divisão por zero
    return TP / (TP + FN)

def especificidade(y_true, y_pred):
    TN = sum((y_true == -1) & (y_pred == -1))
    FP = sum((y_true == -1) & (y_pred == 1))
    if TN + FP == 0:
        return 0  # Evita divisão por zero
    return TN / (TN + FP)


# Função para criar gráfico de matriz de confusão
def plot_confusion_matrix(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == -1))
    FP = sum((y_true == -1) & (y_pred == 1))
    TN = sum((y_true == -1) & (y_pred == -1))
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Classe Negativa", "Classe Positiva"])
    plt.yticks(tick_marks, ["Classe Negativa", "Classe Positiva"])
    plt.xlabel("Previsões")
    plt.ylabel("Rótulos Reais")
    plt.show()

# Treine o Perceptron
pesos_perceptron = train_perceptron(X_treino, Y_treino, 100, 0.1)

# Teste o Perceptron
y_pred_perceptron = [perceptron_test(pesos_perceptron, x) for x in X_teste]

# Calcule as métricas para o Perceptron
accuracy_perceptron = acuracia(Y_teste, y_pred_perceptron)
sensitivity_perceptron = sensibilidade(Y_teste, y_pred_perceptron)
specificity_perceptron = especificidade(Y_teste, y_pred_perceptron)

# Treine o Adaline
pesos_adaline = train_adaline(X_treino, Y_treino, 0.0001, 100, 0.1)

# Teste o Adaline
y_pred_adaline = [adaline_test(pesos_adaline, x) for x in X_teste]

# Calcule as métricas para o Adaline
accuracy_adaline = acuracia(Y_teste, y_pred_adaline)
sensitivity_adaline = sensibilidade(Y_teste, y_pred_adaline)
specificity_adaline = especificidade(Y_teste, y_pred_adaline)

# Determine a melhor e a pior acurácia
best_accuracy_model = 'Perceptron' if accuracy_perceptron > accuracy_adaline else 'Adaline'
worst_accuracy_model = 'Perceptron' if accuracy_perceptron < accuracy_adaline else 'Adaline'

# Crie matrizes de confusão para a melhor e pior acurácia
if best_accuracy_model == 'Perceptron':
    plot_confusion_matrix(Y_teste, y_pred_perceptron)
else:
    plot_confusion_matrix(Y_teste, y_pred_adaline)

if worst_accuracy_model == 'Perceptron':
    plot_confusion_matrix(Y_teste, y_pred_perceptron)
else:
    plot_confusion_matrix(Y_teste, y_pred_adaline)

print(f"Acurácia do Perceptron: {accuracy_perceptron * 100}")
print(f"Acurácia do Adaline: {accuracy_adaline * 100}")

w_perceptron = train_perceptron(X_treino, Y_treino, 100, 0.1)
xx = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
yy_perceptron = -w_perceptron[0]/w_perceptron[2] - w_perceptron[1]/w_perceptron[2] * xx

plt.plot(xx, yy_perceptron, 'k-')
plt.scatter(X[:, 1], X[:, 2], c=Y)
plt.title("Hiperplano de Separação do Perceptron")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot do hiperplano de separação do Adaline
w_adaline = train_adaline(X_treino, Y_treino, 0.0001, 100, 0.1)
yy_adaline = -w_adaline[0]/w_adaline[2] - w_adaline[1]/w_adaline[2] * xx

plt.plot(xx, yy_adaline, 'k-')
plt.scatter(X[:, 1], X[:, 2], c=Y)
plt.title("Hiperplano de Separação do Adaline")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
