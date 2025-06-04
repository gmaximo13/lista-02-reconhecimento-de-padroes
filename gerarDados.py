import numpy as np
import pandas as pd
import os

# Criar DataFrames e adicionar rótulos
def create_dataframe(data, class_label):
    df = pd.DataFrame(data.T, columns=['x1', 'x2'])
    df['class'] = class_label
    return df

# Geração de Dados de Treino
C1_train_data = 1 + 0.2 * np.random.randn(2, 100)
C2_train_data = 2 + 0.2 * np.random.randn(2, 100)
C3_train_data = np.tile(np.array([[0], [2]]), (1, 100)) + 0.2 * np.random.randn(2, 100)

df_C1_train = create_dataframe(C1_train_data, 'C1')
df_C2_train = create_dataframe(C2_train_data, 'C2')
df_C3_train = create_dataframe(C3_train_data, 'C3')
df_train_all = pd.concat([df_C1_train, df_C2_train, df_C3_train], ignore_index=True)

# Geração de Dados de Teste
C1_test_data = 1 + 0.2 * np.random.randn(2, 100)
C2_test_data = 2 + 0.2 * np.random.randn(2, 100)
C3_test_data = np.tile(np.array([[0], [2]]), (1, 100)) + 0.2 * np.random.randn(2, 100)

df_C1_test = create_dataframe(C1_test_data, 'C1')
df_C2_test = create_dataframe(C2_test_data, 'C2')
df_C3_test = create_dataframe(C3_test_data, 'C3')
df_test_all = pd.concat([df_C1_test, df_C2_test, df_C3_test], ignore_index=True)

# Salvamento dos DataFrames em CSV
output_dir = 'data_lista_2'
os.makedirs(output_dir, exist_ok=True)

df_C1_train.to_csv(os.path.join(output_dir, 'C1_train.csv'), index=False)
df_C2_train.to_csv(os.path.join(output_dir, 'C2_train.csv'), index=False)
df_C3_train.to_csv(os.path.join(output_dir, 'C3_train.csv'), index=False)
df_train_all.to_csv(os.path.join(output_dir, 'train_all.csv'), index=False)

df_C1_test.to_csv(os.path.join(output_dir, 'C1_test.csv'), index=False)
df_C2_test.to_csv(os.path.join(output_dir, 'C2_test.csv'), index=False)
df_C3_test.to_csv(os.path.join(output_dir, 'C3_test.csv'), index=False)
df_test_all.to_csv(os.path.join(output_dir, 'test_all.csv'), index=False)

print(f"Dados de treino e teste gerados e salvos na pasta '{output_dir}/'.")