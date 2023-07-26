'''
Usando o dataset da íris, só com sepal e petal length, classes setosa (-1) e versicolor (+1),
criar uma função que recebe X, y, w_inicial, b_inicial e random_state, embaralhando
a cada época caso seja pedido com numpy.random.shuffle(), inicializando com o valor
de random_state (numpy.random.seed(random_state)) uma única vez no início da função.
Os valores de w e b devem ser atualizados cada vez que um ponto da amostra estiver 
classificado errado até que não haja mais erros.
O programa ao final deverá imprimir uma lista com o número de atualizações em cada 
época, os valores de w e b da fronteira de decisão.
obs: embaralhe X e y juntos se não vai ficar tudo errado. 
Melhor embaralhar os números dos índices ao invés de embaralhar o próprio dataset in place.

Exemplo de execução:

digite w0: 1,1
digite b0: 1
random_state: False
[2, 2, 3, 1, 0]
[-3.3 8.1] -1.0

'''
# seu código aqui:
# importar módulos numpy e pandas

import pandas as pd
import numpy as np

# importar o dataset da íris e armazenar nas variáveis X e y o que for necessário

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
    
X = df.iloc[0:100, [0, 2]].values

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

indices = np.arange(X.shape[0])    

# definir a função do perceptron

def perceptron (w_inicial, b_inicial, random_state):
  
  if (random_state != False):
    np.random.seed(random_state)

  n_atts = []
  w = w_inicial
  b = b_inicial
  fica = True
  while fica:
    fica = False
    k = 0
    if (random_state != False):
      np.random.shuffle(indices)
    for i in range(X.shape[0]):
      if (np.dot(w, X[indices[i]]) + b) * y[indices[i]] < 0:
        w = w + y[indices[i]] * X[indices[i]]
        b = b + y[indices[i]]
        k = k+1
        fica = True
    n_atts.append(k)
  return n_atts, w, b

# pedir para o usuário inserir os dados

w0 = np.array(list(map(float, input("digite w0: ").split(","))))


b0 = float(input("digite b0: "))


random_state = input("random_state: ")
if (random_state == "False"):
    random_state = False
else:
    random_state = int(random_state)


# calcular os resultados usando a função

n_atts, w, b = perceptron(w0, b0, random_state)

# imprimir resultados

print(n_atts)
print(w, b)