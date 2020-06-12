import pandas as pd

#separando em previsores e classe
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#separando em treinamento e teste 
from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores,classe,test_size = 0.25 )

#criando a rede neural
import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
#(n de entradas + n de neuronios na saida)/2
#(30+1)/2 = 15.5
classificador.add(Dense(units = 16, activation = 'relu',
  kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 16, activation = 'relu',
  kernel_initializer = 'random_uniform'))


#camada de saída:
classificador.add(Dense(units = 1, activation ='sigmoid'))

#configurar gradiente,erro,etc
classificador.compile(optimizer = 'adam', loss ='binary_crossentropy',
                      metrics = ['binary_accuracy'])

#TREINAMENTO:

#gerar algoritmo de treinamento
classificador.fit(previsores_treinamento,classe_treinamento,
                  batch_size=10, epochs = 100)

#aplicar algoritmo na variavel de teste
previsoes = classificador.predict(previsores_teste)
#converter p/ true e false
previsoes = (previsoes > 0.5)

#analisar o resultado com sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#analisar o resultado com keiras
resultado = classificador.evaluate(previsores_teste,classe_teste)
print(resultado)
