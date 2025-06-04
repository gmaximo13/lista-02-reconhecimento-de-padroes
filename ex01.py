import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configurações Iniciais
data_dir = 'data_lista_2'

# Carregar os dados de treino
df_C1_train = pd.read_csv(os.path.join(data_dir, 'C1_train.csv'))
df_C2_train = pd.read_csv(os.path.join(data_dir, 'C2_train.csv'))
df_C3_train = pd.read_csv(os.path.join(data_dir, 'C3_train.csv'))
df_train_all = pd.read_csv(os.path.join(data_dir, 'train_all.csv'))

C1_train = df_C1_train[['x1', 'x2']].values.T
C2_train = df_C2_train[['x1', 'x2']].values.T
C3_train = df_C3_train[['x1', 'x2']].values.T

# Rótulos verdadeiros de treino (0 para C1, 1 para C2, 2 para C3) - usado para treinamento do Perceptron
y_train_true_numeric = np.array([0] * C1_train.shape[1] + [1] * C2_train.shape[1] + [2] * C3_train.shape[1])

# Visualização dos Dados de Treino
plt.figure(figsize=(8, 6))
plt.plot(C1_train[0, :], C1_train[1, :], 'o', label='C1', alpha=0.6)
plt.plot(C2_train[0, :], C2_train[1, :], 'r*', label='C2', alpha=0.6)
plt.plot(C3_train[0, :], C3_train[1, :], 'k+', label='C3', alpha=0.6)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dados de Treino (C1, C2, C3)')
plt.grid(True)
plt.show()

# Calcular as médias de cada classe de treino para Fisher
m1_train = np.mean(C1_train, axis=1).reshape(-1, 1)
m2_train = np.mean(C2_train, axis=1).reshape(-1, 1)
m3_train = np.mean(C3_train, axis=1).reshape(-1, 1)

# Implementação do Discriminante de Fisher (One-vs-Rest)
# Função para calcular S_W para um conjunto de classes
def calculate_Sw(data_classes, means):
    Sw = np.zeros((data_classes[0].shape[0], data_classes[0].shape[0]))
    for i, data_class in enumerate(data_classes):
        # Transforma a classe de (dim, N) para (N, dim) para iterar por amostra
        for x_vec in data_class.T:
            # Garante que mean_vec_flat seja um vetor linha para np.outer
            mean_vec_flat = means[i].flatten()
            Sw += np.outer(x_vec - mean_vec_flat, x_vec - mean_vec_flat)
    return Sw


# Fisher para C1 vs (C2, C3)
X_rest_c1_train = np.hstack((C2_train, C3_train))
m_rest_c1_train = np.mean(X_rest_c1_train, axis=1).reshape(-1, 1)
Sw_c1_vs_rest_fisher = calculate_Sw([C1_train, X_rest_c1_train], [m1_train, m_rest_c1_train])
w_c1_vs_rest_fisher = np.linalg.inv(Sw_c1_vs_rest_fisher) @ (m1_train - m_rest_c1_train)
# O limiar b é o ponto médio das projeções das médias
proj_m1_c1 = (w_c1_vs_rest_fisher.T @ m1_train)[0, 0]
proj_m_rest_c1 = (w_c1_vs_rest_fisher.T @ m_rest_c1_train)[0, 0]
b_c1_vs_rest_fisher = (proj_m1_c1 + proj_m_rest_c1) / 2

# Fisher para C2 vs (C1, C3)
X_rest_c2_train = np.hstack((C1_train, C3_train))
m_rest_c2_train = np.mean(X_rest_c2_train, axis=1).reshape(-1, 1)
Sw_c2_vs_rest_fisher = calculate_Sw([C2_train, X_rest_c2_train], [m2_train, m_rest_c2_train])
w_c2_vs_rest_fisher = np.linalg.inv(Sw_c2_vs_rest_fisher) @ (m2_train - m_rest_c2_train)
proj_m2_c2 = (w_c2_vs_rest_fisher.T @ m2_train)[0, 0]
proj_m_rest_c2 = (w_c2_vs_rest_fisher.T @ m_rest_c2_train)[0, 0]
b_c2_vs_rest_fisher = (proj_m2_c2 + proj_m_rest_c2) / 2

# Fisher para C3 vs (C1, C2)
X_rest_c3_train = np.hstack((C1_train, C2_train))
m_rest_c3_train = np.mean(X_rest_c3_train, axis=1).reshape(-1, 1)
Sw_c3_vs_rest_fisher = calculate_Sw([C3_train, X_rest_c3_train], [m3_train, m_rest_c3_train])
w_c3_vs_rest_fisher = np.linalg.inv(Sw_c3_vs_rest_fisher) @ (m3_train - m_rest_c3_train)
proj_m3_c3 = (w_c3_vs_rest_fisher.T @ m3_train)[0, 0]
proj_m_rest_c3 = (w_c3_vs_rest_fisher.T @ m_rest_c3_train)[0, 0]
b_c3_vs_rest_fisher = (proj_m3_c3 + proj_m_rest_c3) / 2

X_train_perceptron_all = np.vstack((np.hstack((C1_train, C2_train, C3_train)), np.ones(C1_train.shape[1] * 3)))

def train_perceptron(X_data_with_bias, labels, learning_rate=0.01, max_epochs=1000):
    num_features = X_data_with_bias.shape[0]  # Inclui o bias
    w = np.random.randn(num_features, 1)  # Pesos iniciais, incluindo o peso do bias

    for epoch in range(max_epochs):
        errors_in_epoch = 0
        misclassified_indices = []

        for i in range(X_data_with_bias.shape[1]):
            xi = X_data_with_bias[:, i].reshape(-1, 1)  # Amostra com bias
            prediction = np.dot(w.T, xi)[0, 0]

            if labels[i] * prediction <= 0:
                misclassified_indices.append(i)
                errors_in_epoch += 1

        if errors_in_epoch == 0:
            break

        sum_delta_x_x = np.zeros_like(w)
        for idx in misclassified_indices:
            xi = X_data_with_bias[:, idx].reshape(-1, 1)
            sum_delta_x_x += labels[idx] * xi  # Na formulação de w = w + ...

        w = w + learning_rate * sum_delta_x_x

    return w

# Treinar Perceptron para C1 vs (C2, C3)
labels_c1_vs_rest_p = np.array([1] * C1_train.shape[1] + [-1] * C2_train.shape[1] + [-1] * C3_train.shape[1])
w_c1_vs_rest_perceptron = train_perceptron(X_train_perceptron_all, labels_c1_vs_rest_p)

# Treinar Perceptron para C2 vs (C1, C3)
labels_c2_vs_rest_p = np.array([-1] * C1_train.shape[1] + [1] * C2_train.shape[1] + [-1] * C3_train.shape[1])
w_c2_vs_rest_perceptron = train_perceptron(X_train_perceptron_all, labels_c2_vs_rest_p)

# Treinar Perceptron para C3 vs (C1, C2)
labels_c3_vs_rest_p = np.array([-1] * C1_train.shape[1] + [-1] * C2_train.shape[1] + [1] * C3_train.shape[1])
w_c3_vs_rest_perceptron = train_perceptron(X_train_perceptron_all, labels_c3_vs_rest_p)

# Funções Auxiliares para Plotagem
x_min_data = np.min(X_train_perceptron_all[0, :])
x_max_data = np.max(X_train_perceptron_all[0, :])
y_min_data = np.min(X_train_perceptron_all[1, :])
y_max_data = np.max(X_train_perceptron_all[1, :])

x_vals_plot = np.linspace(x_min_data - 0.5, x_max_data + 0.5, 100)

def plot_fisher_line(w_vec, b_val, x_vals, color, linestyle, label):
    if w_vec[1, 0] != 0:
        y_vals = (-w_vec[0, 0] * x_vals + b_val) / w_vec[1, 0]
        plt.plot(x_vals, y_vals, color, linestyle=linestyle, label=label)
    else:
        plt.axvline(x=b_val / w_vec[0, 0], color=color, linestyle=linestyle, label=label)

def plot_perceptron_line_from_w(w_perceptron, x_vals, color, linestyle, label):
    if w_perceptron[1, 0] != 0:
        y_vals = (-w_perceptron[0, 0] * x_vals - w_perceptron[2, 0]) / w_perceptron[1, 0]
        plt.plot(x_vals, y_vals, color, linestyle=linestyle, label=label)
    else:  # Linha vertical
        plt.axvline(x=-w_perceptron[2, 0] / w_perceptron[0, 0], color=color, linestyle=linestyle, label=label)

# Gráfico 1: Discriminante de Fisher (One-vs-Rest)
plt.figure(figsize=(10, 8))
plt.scatter(C1_train[0, :], C1_train[1, :], c='blue', marker='o', label='C1 (Treino)', alpha=0.6)
plt.scatter(C2_train[0, :], C2_train[1, :], c='red', marker='*', label='C2 (Treino)', alpha=0.6)
plt.scatter(C3_train[0, :], C3_train[1, :], c='green', marker='+', label='C3 (Treino)', alpha=0.6)

plot_fisher_line(w_c1_vs_rest_fisher, b_c1_vs_rest_fisher, x_vals_plot, 'blue', '-', 'Fisher: C1 vs Rest')
plot_fisher_line(w_c2_vs_rest_fisher, b_c2_vs_rest_fisher, x_vals_plot, 'red', '-', 'Fisher: C2 vs Rest')
plot_fisher_line(w_c3_vs_rest_fisher, b_c3_vs_rest_fisher, x_vals_plot, 'green', '-', 'Fisher: C3 vs Rest')

plt.title('Classificador Discriminante de Fisher (One-vs-Rest)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data - 0.5, x_max_data + 0.5)
plt.ylim(y_min_data - 0.5, y_max_data + 0.5)
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 2: Perceptron (One-vs-Rest)
plt.figure(figsize=(10, 8))
plt.scatter(C1_train[0, :], C1_train[1, :], c='blue', marker='o', label='C1 (Treino)', alpha=0.6)
plt.scatter(C2_train[0, :], C2_train[1, :], c='red', marker='*', label='C2 (Treino)', alpha=0.6)
plt.scatter(C3_train[0, :], C3_train[1, :], c='green', marker='+', label='C3 (Treino)', alpha=0.6)

plot_perceptron_line_from_w(w_c1_vs_rest_perceptron, x_vals_plot, 'blue', '-', 'Perceptron: C1 vs Rest')
plot_perceptron_line_from_w(w_c2_vs_rest_perceptron, x_vals_plot, 'red', '-', 'Perceptron: C2 vs Rest')
plot_perceptron_line_from_w(w_c3_vs_rest_perceptron, x_vals_plot, 'green', '-', 'Perceptron: C3 vs Rest')

plt.title('Classificador Perceptron (One-vs-Rest)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data - 0.5, x_max_data + 0.5)
plt.ylim(y_min_data - 0.5, y_max_data + 0.5)
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 3: Comparativo Fisher vs Perceptron
plt.figure(figsize=(12, 10))
plt.scatter(C1_train[0, :], C1_train[1, :], c='blue', marker='o', label='C1 (Treino)', alpha=0.6)
plt.scatter(C2_train[0, :], C2_train[1, :], c='red', marker='*', label='C2 (Treino)', alpha=0.6)
plt.scatter(C3_train[0, :], C3_train[1, :], c='green', marker='+', label='C3 (Treino)', alpha=0.6)

# Fronteiras do Fisher
plot_fisher_line(w_c1_vs_rest_fisher, b_c1_vs_rest_fisher, x_vals_plot, 'blue', ':', 'Fisher: C1 vs Rest')
plot_fisher_line(w_c2_vs_rest_fisher, b_c2_vs_rest_fisher, x_vals_plot, 'red', ':', 'Fisher: C2 vs Rest')
plot_fisher_line(w_c3_vs_rest_fisher, b_c3_vs_rest_fisher, x_vals_plot, 'green', ':', 'Fisher: C3 vs Rest')

# Fronteiras do Perceptron
plot_perceptron_line_from_w(w_c1_vs_rest_perceptron, x_vals_plot, 'blue', '--', 'Perceptron: C1 vs Rest')
plot_perceptron_line_from_w(w_c2_vs_rest_perceptron, x_vals_plot, 'red', '--', 'Perceptron: C2 vs Rest')
plot_perceptron_line_from_w(w_c3_vs_rest_perceptron, x_vals_plot, 'green', '--', 'Perceptron: C3 vs Rest')

plt.title('Comparativo: Fisher (Pontilhado) vs Perceptron (Tracejado) (OvR)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data - 0.5, x_max_data + 0.5)
plt.ylim(y_min_data - 0.5, y_max_data + 0.5)
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Parâmetros Projetados ---")
print("Discriminante de Fisher (One-vs-Rest):")
print(f"  C1 vs Rest: w = {w_c1_vs_rest_fisher.flatten()}, b = {b_c1_vs_rest_fisher:.4f}")
print(f"  C2 vs Rest: w = {w_c2_vs_rest_fisher.flatten()}, b = {b_c2_vs_rest_fisher:.4f}")
print(f"  C3 vs Rest: w = {w_c3_vs_rest_fisher.flatten()}, b = {b_c3_vs_rest_fisher:.4f}")
print("\nPerceptron (One-vs-Rest):")
print(f"  C1 vs Rest (w1, w2, w0): {w_c1_vs_rest_perceptron.flatten()}")
print(f"  C2 vs Rest (w1, w2, w0): {w_c2_vs_rest_perceptron.flatten()}")
print(f"  C3 vs Rest (w1, w2, w0): {w_c3_vs_rest_perceptron.flatten()}")