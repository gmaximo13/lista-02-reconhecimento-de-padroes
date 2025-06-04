# Lista 02 – Reconhecimento de Padrões

Este repositório contém os códigos, gráficos e análises desenvolvidas para a **Lista 02 da disciplina GAT117 - Reconhecimento de Padrões** da Universidade Federal de Lavras (UFLA), ministrada em 2025.

## 👨‍🎓 Autor
**Gustavo dos Santos Moreira Máximo**

## 🧠 Objetivo

Esta atividade tem como objetivo aplicar e comparar diferentes **algoritmos de classificação multi-classe**, estendendo os conceitos de classificadores binários. Os algoritmos abordados incluem:

- **Discriminante de Fisher** (abordagem One-vs-Rest)
- **Perceptron** (abordagem One-vs-Rest)

---

## ⚙️ Execução dos Códigos

Para executar os códigos e reproduzir os resultados, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/gmaximo13/lista-02-reconhecimento-de-padroes.git
    ```

2.  **Instale as dependências:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

3.  **Execute os scripts em ordem:**

    * **Primeiro, gere os dados:**
        ```bash
        python gerar_dados_lista2.py
        ```
        Este script criará a pasta `data_lista_2/` e salvará os arquivos CSV de treino e teste nela.

    * **Em seguida, projete os classificadores (Questão 1):**
        ```bash
        python resolver_lista2_q1.py
        ```
        Este script irá carregar os dados de treino gerados, projetar os classificadores Discriminante de Fisher e Perceptron (One-vs-Rest), e exibir os gráficos de projeto, além de imprimir os parâmetros (pesos e limiares) obtidos.

    * **Por fim, teste e avalie os classificadores (Questão 2):**
        ```bash
        python resolver_lista2_q2_q3.py
        ```
        Este script irá carregar os dados de teste e utilizar os parâmetros **hardcoded** dos classificadores obtidos na Questão 1 (os que foram impressos ao final do script `resolver_lista2_q1.py`). Ele exibirá a matriz de confusão, métricas de desempenho e gráficos das fronteiras sobre os dados de teste.

---

## 🧪 Descrição dos Experimentos

Os experimentos da Lista 02 focam na extensão de classificadores lineares para problemas multi-classe (3 classes) utilizando a estratégia One-vs-Rest (OvR), e na avaliação de seu desempenho em dados não vistos.

| Questão | Classificador / Tópico                           | Dados Usados           | Observações                                                                 |
|--------:|------------------------------------------------|-------------------------|------------------------------------------------------------------------------|
|       1 | Discriminante de Fisher (OvR) e Perceptron (OvR) | Dados de Treino (100 amostras/classe) | Projeto dos classificadores multi-classe e visualização das fronteiras de decisão em relação aos dados de treino. O Perceptron utilizado é baseado no "Algoritmo Perceptron" do Slide 12 da Aula 3. |
|       2 | Teste dos classificadores da Questão 1           | Dados de Teste (100 amostras/classe)  | Avaliação do desempenho dos classificadores em dados não vistos, com Matrizes de Confusão, métricas de acerto por classe (Recall, Precisão) e Acurácia Geral. Inclui gráficos das fronteiras sobre os dados de teste.   |

---

## 📊 Comparação dos Algoritmos

A análise dos resultados, tanto nos dados de treino quanto de teste, demonstrou as características de cada classificador:

-   O **Discriminante de Fisher (OvR)** apresentou **excelente desempenho**, mantendo 100% de acerto nas classes mesmo em dados de teste. Isso é esperado, pois para dados linearmente separáveis com distribuições Gaussianas e covariâncias semelhantes, o Fisher se aproxima da solução ótima teórica. Suas fronteiras de decisão, que consideram a dispersão e centralidade das classes, tendem a ser mais robustas.
-   O **Perceptron (OvR)** também alcançou um **alto desempenho**, inclusive 100% de acurácia geral nos dados de teste. Apesar de suas fronteiras de decisão poderem parecer "menos ideais" visualmente em comparação com o Fisher (já que o Perceptron busca *qualquer* hiperplano que separe os pontos de treino, e não a fronteira ótima probabilisticamente), ele conseguiu classificar todos os pontos de teste corretamente para este conjunto de dados linearmente separável.

## 📌 Conclusão

Este estudo aprofundou a aplicação de classificadores lineares para problemas multi-classe, utilizando a estratégia One-vs-Rest (OvR). Ficou evidente que, para conjuntos de dados linearmente separáveis e bem definidos como os utilizados, tanto o Discriminante de Fisher quanto o Perceptron são capazes de projetar fronteiras de decisão eficazes, com o Fisher oferecendo uma solução mais robusta do ponto de vista teórico. A avaliação em dados de teste é crucial para confirmar a capacidade de generalização dos modelos.

---
