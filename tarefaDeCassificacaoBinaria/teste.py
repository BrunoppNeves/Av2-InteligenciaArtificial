import numpy as np
import matplotlib.pyplot as plt

# Carregando dados
dados = np.loadtxt('tarefaDeCassificacaoBinaria/DataAV2.csv', delimiter=',')
X = dados[:, 0:2]
Y = dados[:, 2]
N, p = X.shape

plt.scatter(X[:, 0], X[:, 1], c=Y, linewidths=0.4, edgecolors='k')
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
                weights = weights + learning_rate * \
                    (target - y_t) * input_sample
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


accuracies_perceptron = []
sensitivities_perceptron = []
specificities_perceptron = []
accuracies_adaline = []
sensitivities_adaline = []
specificities_adaline = []

best_accuracy_perceptron = -1
worst_accuracy_perceptron = 2
best_accuracy_adaline = -1
worst_accuracy_adaline = 2

Y_teste_best_accuracy_perceptron = []
X_teste_best_accuracy_perceptron = []

Y_teste_worst_accuracy_perceptron = []
X_teste_worst_accuracy_perceptron = []

Y_teste_best_accuracy_adaline = []
X_teste_best_accuracy_adaline = []

Y_teste_worst_accuracy_adaline = []
X_teste_worst_accuracy_adaline = []

for round in range(10):
    X_treino, Y_treino, X_teste, Y_teste = dividir_dados()

    w_perceptron = train_perceptron(X_treino, Y_treino, 100, 0.8)
    w_adaline = train_adaline(X_treino, Y_treino, 0.001, 100, 0.7)

    y_pred_perceptron = np.zeros(len(X_teste))
    y_pred_adaline = np.zeros(len(X_teste))

    for i in range(len(X_teste)):
        y_pred_perceptron[i] = perceptron_test(w_perceptron, X_teste[i])
        y_pred_adaline[i] = adaline_test(w_adaline, X_teste[i])

    accuracy_perceptron = acuracia(Y_teste, y_pred_perceptron)
    accuracy_adaline = acuracia(Y_teste, y_pred_adaline)

    accuracies_perceptron.append(accuracy_perceptron)
    accuracies_adaline.append(accuracy_adaline)

    sensitivity_perceptron = sensibilidade(Y_teste, y_pred_perceptron)
    sensitivity_adaline = sensibilidade(Y_teste, y_pred_adaline)

    sensitivities_perceptron.append(sensitivity_perceptron)
    sensitivities_adaline.append(sensitivity_adaline)

    specificity_perceptron = especificidade(Y_teste, y_pred_perceptron)
    specificity_adaline = especificidade(Y_teste, y_pred_adaline)

    specificities_perceptron.append(specificity_perceptron)
    specificities_adaline.append(specificity_adaline)

    best_accuracy_perceptron = max(
        best_accuracy_perceptron, accuracy_perceptron)
    worst_accuracy_perceptron = min(
        worst_accuracy_perceptron, accuracy_perceptron)
    best_accuracy_adaline = max(best_accuracy_adaline, accuracy_adaline)
    worst_accuracy_adaline = min(worst_accuracy_adaline, accuracy_adaline)

    if accuracy_perceptron == best_accuracy_perceptron:
        Y_teste_best_accuracy_perceptron = Y_teste
        X_teste_best_accuracy_perceptron = X_teste
    if accuracy_perceptron == worst_accuracy_perceptron:
        Y_teste_worst_accuracy_perceptron = Y_teste
        X_teste_worst_accuracy_perceptron = X_teste
    if accuracy_adaline == best_accuracy_adaline:
        Y_teste_best_accuracy_adaline = Y_teste
        X_teste_best_accuracy_adaline = X_teste
    if accuracy_adaline == worst_accuracy_adaline:
        Y_teste_worst_accuracy_adaline = Y_teste
        X_teste_worst_accuracy_adaline = X_teste


# Calculate statistics
mean_accuracy_perceptron = np.mean(accuracies_perceptron)
std_accuracy_perceptron = np.std(accuracies_perceptron)
max_accuracy_perceptron = max(accuracies_perceptron)
min_accuracy_perceptron = min(accuracies_perceptron)

mean_sensitivity_perceptron = np.mean(sensitivities_perceptron)
std_sensitivity_perceptron = np.std(sensitivities_perceptron)
max_sensitivity_perceptron = max(sensitivities_perceptron)
min_sensitivity_perceptron = min(sensitivities_perceptron)

mean_specificity_perceptron = np.mean(specificities_perceptron)
std_specificity_perceptron = np.std(specificities_perceptron)
max_specificity_perceptron = max(specificities_perceptron)
min_specificity_perceptron = min(specificities_perceptron)

mean_accuracy_adaline = np.mean(accuracies_adaline)
std_accuracy_adaline = np.std(accuracies_adaline)
max_accuracy_adaline = max(accuracies_adaline)
min_accuracy_adaline = min(accuracies_adaline)

mean_sensitivity_adaline = np.mean(sensitivities_adaline)
std_sensitivity_adaline = np.std(sensitivities_adaline)
max_sensitivity_adaline = max(sensitivities_adaline)
min_sensitivity_adaline = min(sensitivities_adaline)

mean_specificity_adaline = np.mean(specificities_adaline)
std_specificity_adaline = np.std(specificities_adaline)
max_specificity_adaline = max(specificities_adaline)
min_specificity_adaline = min(specificities_adaline)

print("Perceptron Results:")
print(f"Mean Accuracy: {mean_accuracy_perceptron * 100}")
print(f"Desvio padrao: {std_accuracy_perceptron * 100}")
print(f"Max Accuracy: {max_accuracy_perceptron * 100}")
print(f"Min Accuracy: {min_accuracy_perceptron * 100}")
print("")
print(f"Mean Sensitivity: {mean_sensitivity_perceptron}")
print(f"Standard Deviation Sensitivity: {std_sensitivity_perceptron}")
print(f"Max Sensitivity: {max_sensitivity_perceptron}")
print(f"Min Sensitivity: {min_sensitivity_perceptron}")
print("")
print(f"Mean Specificity: {mean_specificity_perceptron}")
print(f"Standard Deviation Specificity: {std_specificity_perceptron}")
print(f"Max Specificity: {max_specificity_perceptron}")
print(f"Min Specificity: {min_specificity_perceptron}")
print("")
print("Adaline Results:")
print(f"Mean Accuracy: {mean_accuracy_adaline * 100}")
print(f"Standard Deviation: {std_accuracy_adaline * 100}")
print(f"Max Accuracy: {max_accuracy_adaline * 100}")
print(f"Min Accuracy: {min_accuracy_adaline * 100}")
print("")
print(f"Mean Sensitivity: {mean_sensitivity_adaline}")
print(f"Standard Deviation Sensitivity: {std_sensitivity_adaline}")
print(f"Max Sensitivity: {max_sensitivity_adaline}")
print(f"Min Sensitivity: {min_sensitivity_adaline}")
print("")
print(f"Mean Specificity: {mean_specificity_adaline}")
print(f"Standard Deviation Specificity: {std_specificity_adaline}")
print(f"Max Specificity: {max_specificity_adaline}")
print(f"Min Specificity: {min_specificity_adaline}")

# Plot confusion matrix best and worst accuracy perceptron
plot_confusion_matrix(Y_teste_best_accuracy_perceptron, y_pred_perceptron)
plot_confusion_matrix(Y_teste_worst_accuracy_perceptron, y_pred_perceptron)

# Plot confusion matrix best and worst accuracy adaline
plot_confusion_matrix(Y_teste_best_accuracy_adaline, y_pred_adaline)
plot_confusion_matrix(Y_teste_worst_accuracy_adaline, y_pred_adaline)

# Melhor caso
plt.scatter(X_teste_best_accuracy_perceptron[:, 1], X_teste_best_accuracy_perceptron[:,
            2], c=Y_teste_best_accuracy_perceptron, linewidths=0.4, edgecolors='k')
plt.title("Gráfico de Dispersão (Melhor caso perceptron) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Pior caso
plt.scatter(X_teste_worst_accuracy_perceptron[:, 1], X_teste_worst_accuracy_perceptron[:,
            2], c=Y_teste_worst_accuracy_perceptron, linewidths=0.4, edgecolors='k')
plt.title("Gráfico de Dispersão (Pior caso perceptron) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Melhor caso
plt.scatter(X_teste_best_accuracy_adaline[:, 1], X_teste_best_accuracy_adaline[:,
            2], c=Y_teste_best_accuracy_adaline, linewidths=0.4, edgecolors='k')
plt.title("Gráfico de Dispersão (Melhor caso adaline) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Pior caso
plt.scatter(X_teste_worst_accuracy_adaline[:, 1], X_teste_worst_accuracy_adaline[:,
            2], c=Y_teste_worst_accuracy_adaline, linewidths=0.4, edgecolors='k')
plt.title("Gráfico de Dispersão (Pior caso adaline) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
