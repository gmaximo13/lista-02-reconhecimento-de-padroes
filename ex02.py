import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os

data_dir = 'data_lista_2'

df_C1_test = pd.read_csv(os.path.join(data_dir, 'C1_test.csv'))
df_C2_test = pd.read_csv(os.path.join(data_dir, 'C2_test.csv'))
df_C3_test = pd.read_csv(os.path.join(data_dir, 'C3_test.csv'))
df_test_all = pd.read_csv(os.path.join(data_dir, 'test_all.csv'))

C1_test = df_C1_test[['x1', 'x2']].values.T
C2_test = df_C2_test[['x1', 'x2']].values.T
C3_test = df_C3_test[['x1', 'x2']].values.T

# Rótulos verdadeiros de teste (0 para C1, 1 para C2, 2 para C3)
y_test_true = np.array([0] * C1_test.shape[1] + [1] * C2_test.shape[1] + [2] * C3_test.shape[1])

# Preparar dados de teste para os classificadores
X_test_plot = np.hstack((C1_test, C2_test, C3_test))
X_test_fisher = np.hstack((C1_test, C2_test, C3_test))
X_test_perceptron = np.vstack((X_test_fisher, np.ones(X_test_fisher.shape[1])))

# Visualização dos Dados de Teste
print("Visualizando Dados de Teste...")
plt.figure(figsize=(8, 6))
plt.plot(C1_test[0, :], C1_test[1, :], 'o', label='C1', alpha=0.6)
plt.plot(C2_test[0, :], C2_test[1, :], 'r*', label='C2', alpha=0.6)
plt.plot(C3_test[0, :], C3_test[1, :], 'k+', label='C3', alpha=0.6)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Dados de Teste (C1, C2, C3)')
plt.grid(True)
plt.show()

print("\n" + "=" * 50 + "\n")

print("Questão 2: Avaliação dos Classificadores em Dados de Teste\n")

# Resultados obtidos no ex01
# Discriminante de Fisher (One-vs-Rest):
#   C1 vs Rest: w = [-0.00026527 -0.08130789], b = -0.1210
#   C2 vs Rest: w = [0.09247654 0.08055185], b = 0.2541
#   C3 vs Rest: w = [-0.09004443  0.08138488], b = 0.0760
#
# Perceptron (One-vs-Rest):
#   C1 vs Rest (w1, w2, w0): [-0.13256431 -2.39356268  3.43997807]
#   C2 vs Rest (w1, w2, w0): [ 2.46244676  0.06643738 -3.81644277]
#   C3 vs Rest (w1, w2, w0): [-3.48712496  1.52223907  -0.95470794]

w_c1_vs_rest_fisher = np.array([[-0.00026527], [-0.08130789]])
b_c1_vs_rest_fisher = -0.1210

w_c2_vs_rest_fisher = np.array([[0.09247654], [0.08055185]])
b_c2_vs_rest_fisher = 0.2541

w_c3_vs_rest_fisher = np.array([[-0.09004443], [0.08138488]])
b_c3_vs_rest_fisher = 0.0760

# Perceptron (One-vs-Rest):
w_c1_vs_rest_perceptron = np.array([[-0.13256431], [-2.39356268], [3.43997807]])
w_c2_vs_rest_perceptron = np.array([[2.46244676], [0.06643738], [-3.81644277]])
w_c3_vs_rest_perceptron = np.array([[-3.48712496], [1.52223907], [-0.95470794]])

# Funções Auxiliares para Plotagem
x_min_data_plot = np.min(X_test_plot[0, :])
x_max_data_plot = np.max(X_test_plot[0, :])
y_min_data_plot = np.min(X_test_plot[1, :])
y_max_data_plot = np.max(X_test_plot[1, :])

x_vals_plot_range = np.linspace(x_min_data_plot - 0.5, x_max_data_plot + 0.5, 100)

def plot_fisher_line(w_vec, b_val, x_vals, color, linestyle, label):
    if w_vec[1, 0] != 0:
        y_vals = (-w_vec[0, 0] * x_vals + b_val) / w_vec[1, 0]
        plt.plot(x_vals, y_vals, color, linestyle=linestyle, label=label)
    else:  # Linha vertical (w[1] é zero ou muito próximo de zero)
        plt.axvline(x=b_val / w_vec[0, 0], color=color, linestyle=linestyle, label=label)  # x = b/w1

def plot_perceptron_line_from_w(w_perceptron, x_vals, color, linestyle, label):
    if w_perceptron[1, 0] != 0:
        y_vals = (-w_perceptron[0, 0] * x_vals - w_perceptron[2, 0]) / w_perceptron[1, 0]
        plt.plot(x_vals, y_vals, color, linestyle=linestyle, label=label)
    else:  # Linha vertical
        plt.axvline(x=-w_perceptron[2, 0] / w_perceptron[0, 0], color=color, linestyle=linestyle, label=label)

# Predição e Avaliação do Discriminante de Fisher (One-vs-Rest)
def predict_fisher_ovr_multi_class(X_data, w_c1, b_c1, w_c2, b_c2, w_c3, b_c3):
    predictions = []
    for x in X_data.T:
        x_reshaped = x.reshape(-1, 1)
        g1_val = (w_c1.T @ x_reshaped)[0, 0] - b_c1
        g2_val = (w_c2.T @ x_reshaped)[0, 0] - b_c2
        g3_val = (w_c3.T @ x_reshaped)[0, 0] - b_c3

        scores = [g1_val, g2_val, g3_val]
        predicted_class = np.argmax(scores)
        predictions.append(predicted_class)
    return np.array(predictions)

y_pred_fisher = predict_fisher_ovr_multi_class(
    X_test_fisher,
    w_c1_vs_rest_fisher, b_c1_vs_rest_fisher,
    w_c2_vs_rest_fisher, b_c2_vs_rest_fisher,
    w_c3_vs_rest_fisher, b_c3_vs_rest_fisher
)

print("--- Resultados para o Discriminante de Fisher (One-vs-Rest) ---")
cm_fisher = confusion_matrix(y_test_true, y_pred_fisher)
print("Matriz de Confusão:")
print(cm_fisher)

labels_str = ['C1', 'C2', 'C3']
for i, label in enumerate(labels_str):
    tp = cm_fisher[i, i]
    fn = np.sum(cm_fisher[i, :]) - tp
    fp = np.sum(cm_fisher[:, i]) - tp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\nClasse {label}:")
    print(f"  Percentual de Acerto (Recall): {recall * 100:.2f}%")
    print(f"  Precisão: {precision:.4f}")

accuracy_fisher_test = accuracy_score(y_test_true, y_pred_fisher)
print(f"\nAcurácia Geral (Fisher): {accuracy_fisher_test * 100:.2f}%")

# Gráfico 1: Discriminante de Fisher (One-vs-Rest) com Dados de Teste
plt.figure(figsize=(10, 8))
plt.scatter(C1_test[0, :], C1_test[1, :], c='blue', marker='o', label='C1 (Teste)', alpha=0.6)
plt.scatter(C2_test[0, :], C2_test[1, :], c='red', marker='*', label='C2 (Teste)', alpha=0.6)
plt.scatter(C3_test[0, :], C3_test[1, :], c='green', marker='+', label='C3 (Teste)', alpha=0.6)

plot_fisher_line(w_c1_vs_rest_fisher, b_c1_vs_rest_fisher, x_vals_plot_range, 'blue', '-', 'Fisher: C1 vs Rest')
plot_fisher_line(w_c2_vs_rest_fisher, b_c2_vs_rest_fisher, x_vals_plot_range, 'red', '-', 'Fisher: C2 vs Rest')
plot_fisher_line(w_c3_vs_rest_fisher, b_c3_vs_rest_fisher, x_vals_plot_range, 'green', '-', 'Fisher: C3 vs Rest')

plt.title('Classificador Discriminante de Fisher (OvR) em Dados de Teste')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data_plot - 0.5, x_max_data_plot + 0.5)
plt.ylim(y_min_data_plot - 0.5, y_max_data_plot + 0.5)
plt.legend()
plt.grid(True)
plt.show()

# Predição e Avaliação do Perceptron (One-vs-Rest)
def predict_perceptron_ovr_multi_class(X_data_with_bias, w_c1, w_c2, w_c3):
    predictions = []
    for x_with_bias in X_data_with_bias.T:
        x_reshaped = x_with_bias.reshape(-1, 1)
        output_c1 = (w_c1.T @ x_reshaped)[0, 0]
        output_c2 = (w_c2.T @ x_reshaped)[0, 0]
        output_c3 = (w_c3.T @ x_reshaped)[0, 0]

        scores = [output_c1, output_c2, output_c3]
        predicted_class = np.argmax(scores)
        predictions.append(predicted_class)
    return np.array(predictions)


y_pred_perceptron = predict_perceptron_ovr_multi_class(
    X_test_perceptron,
    w_c1_vs_rest_perceptron, w_c2_vs_rest_perceptron, w_c3_vs_rest_perceptron
)

print("\n--- Resultados para o Perceptron (One-vs-Rest) ---")
cm_perceptron = confusion_matrix(y_test_true, y_pred_perceptron)
print("Matriz de Confusão:")
print(cm_perceptron)

for i, label in enumerate(labels_str):
    tp = cm_perceptron[i, i]
    fn = np.sum(cm_perceptron[i, :]) - tp
    fp = np.sum(cm_perceptron[:, i]) - tp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\nClasse {label}:")
    print(f"  Percentual de Acerto (Recall): {recall * 100:.2f}%")
    print(f"  Precisão: {precision:.4f}")

accuracy_perceptron_test = accuracy_score(y_test_true, y_pred_perceptron)
print(f"\nAcurácia Geral (Perceptron): {accuracy_perceptron_test * 100:.2f}%")

# Gráfico 2: Perceptron (One-vs-Rest) com Dados de Teste
plt.figure(figsize=(10, 8))
plt.scatter(C1_test[0, :], C1_test[1, :], c='blue', marker='o', label='C1 (Teste)', alpha=0.6)
plt.scatter(C2_test[0, :], C2_test[1, :], c='red', marker='*', label='C2 (Teste)', alpha=0.6)
plt.scatter(C3_test[0, :], C3_test[1, :], c='green', marker='+', label='C3 (Teste)', alpha=0.6)

plot_perceptron_line_from_w(w_c1_vs_rest_perceptron, x_vals_plot_range, 'blue', '-', 'Perceptron: C1 vs Rest')
plot_perceptron_line_from_w(w_c2_vs_rest_perceptron, x_vals_plot_range, 'red', '-', 'Perceptron: C2 vs Rest')
plot_perceptron_line_from_w(w_c3_vs_rest_perceptron, x_vals_plot_range, 'green', '-', 'Perceptron: C3 vs Rest')

plt.title('Classificador Perceptron (OvR) em Dados de Teste')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data_plot - 0.5, x_max_data_plot + 0.5)
plt.ylim(y_min_data_plot - 0.5, y_max_data_plot + 0.5)
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 3: Comparativo Fisher vs Perceptron (One-vs-Rest) com Dados de Teste
plt.figure(figsize=(12, 10))
plt.scatter(C1_test[0, :], C1_test[1, :], c='blue', marker='o', label='C1 (Teste)', alpha=0.6)
plt.scatter(C2_test[0, :], C2_test[1, :], c='red', marker='*', label='C2 (Teste)', alpha=0.6)
plt.scatter(C3_test[0, :], C3_test[1, :], c='green', marker='+', label='C3 (Teste)', alpha=0.6)

# Fronteiras do Fisher
plot_fisher_line(w_c1_vs_rest_fisher, b_c1_vs_rest_fisher, x_vals_plot_range, 'blue', ':', 'Fisher: C1 vs Rest')
plot_fisher_line(w_c2_vs_rest_fisher, b_c2_vs_rest_fisher, x_vals_plot_range, 'red', ':', 'Fisher: C2 vs Rest')
plot_fisher_line(w_c3_vs_rest_fisher, b_c3_vs_rest_fisher, x_vals_plot_range, 'green', ':', 'Fisher: C3 vs Rest')

# Fronteiras do Perceptron
plot_perceptron_line_from_w(w_c1_vs_rest_perceptron, x_vals_plot_range, 'blue', '--', 'Perceptron: C1 vs Rest')
plot_perceptron_line_from_w(w_c2_vs_rest_perceptron, x_vals_plot_range, 'red', '--', 'Perceptron: C2 vs Rest')
plot_perceptron_line_from_w(w_c3_vs_rest_perceptron, x_vals_plot_range, 'green', '--', 'Perceptron: C3 vs Rest')

plt.title('Comparativo: Fisher (Pontilhado) vs Perceptron (Tracejado) (OvR) em Dados de Teste')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(x_min_data_plot - 0.5, x_max_data_plot + 0.5)
plt.ylim(y_min_data_plot - 0.5, y_max_data_plot + 0.5)
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 50 + "\n")