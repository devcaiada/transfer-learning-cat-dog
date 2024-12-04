# Treinamento de Rede Neural com Transfer Learning

Este código utiliza Transfer Learning com a arquitetura pré-treinada VGG16 para classificar imagens de cães e gatos. O código inclui o carregamento dos dados, a construção do modelo, o treinamento inicial, o fine-tuning, a geração de gráficos de precisão e perda, e a exportação do modelo treinado.

**Dataset utilizado**: [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

![cat-n-dog](https://github.com/devcaiada/transfer-learning-cat-dog/blob/main/assets/cat_n_dogs.png?raw=true)

## Dependências

- TensorFlow

- Keras (integrado ao TensorFlow)

- Matplotlib

Para instalar as dependências, execute no terminal:

```python
pip install tensorflow matplotlib
```

## Estrutura do Código

### 1. Importação

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
```

### 2. Definição dos Diretórios do Dataset

```python
train_dir = r'c:\Users\user\Downloads\cats_and_dogs_small\train'
validation_dir = r'c:\Users\user\Downloads\cats_and_dogs_small\validation'
```

### 3. Pré-processamento dos Dados

Utilizamos **ImageDataGenerator** para aumentar e normalizar as imagens.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

### 4. Construção do Modelo

Carregamos o modelo VGG16 pré-treinado, removendo a camada superior e adicionando camadas personalizadas.

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

![cat-vs-dog](https://github.com/devcaiada/transfer-learning-cat-dog/blob/main/assets/cats_vs_dogs.png?raw=true)

### 5. Congelamento das Camadas Base

Congelamos as camadas do modelo base VGG16 para treinar apenas as novas camadas adicionadas.

```python
for layer in base_model.layers:
    layer.trainable = False
```

### 6. Compilação e Treinamento do Modelo Inicial

Compilamos e treinamos o modelo com as camadas congeladas.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### 7. Fine-Tuning

Descongelamos algumas das últimas camadas do VGG16 e continuamos o treinamento com uma taxa de aprendizado menor.

```python
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
```

### 8. Salvar o Modelo Treinado

Salvamos o modelo treinado em um arquivo **.h5**.

```python
model.save('cats_and_dogs_classifier.h5')
```

### 9. Plotar Gráficos de Precisão e Perda

Definimos uma função para plotar gráficos de precisão e perda para o treinamento e validação.

```python
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

plot_history(history)
plot_history(history_fine)
```

## Considerações Finais

Este código implementa uma abordagem de Transfer Learning usando a arquitetura VGG16 pré-treinada para classificar imagens de cães e gatos, utilizando o "Kaggle Cats and Dogs Dataset". Esta abordagem é poderosa porque aproveita os recursos previamente aprendidos pela VGG16 em um grande conjunto de dados (ImageNet), o que pode melhorar significativamente a precisão e eficiência do modelo em comparação com o treinamento do zero.

### Resultados Esperados

**1. Acurácia do Modelo**: Com um dataset balanceado e um número suficiente de imagens de treino e validação, você pode esperar uma acurácia elevada. Durante o processo de treinamento, é comum observar uma acurácia de validação que se estabiliza após algumas épocas.

**2. Perda (Loss)**: A perda (loss) deve diminuir ao longo das épocas de treinamento. Uma perda menor indica que o modelo está aprendendo bem a distinguir entre as classes de cães e gatos.

**3. Convergência do Modelo**: Durante o fine-tuning, ao descongelar algumas camadas do modelo VGG16, espera-se que a acurácia de validação melhore ainda mais e que a perda diminua. Isso ocorre porque o modelo pode ajustar melhor os parâmetros das camadas pré-treinadas para se adaptar ao novo conjunto de dados específico.

### Gráficos de Acurácia e Perda

Os gráficos gerados durante o treinamento ajudarão a visualizar o desempenho do modelo:

- **Gráfico de Acurácia**: Você verá duas curvas - uma para a acurácia de treinamento e outra para a acurácia de validação. Idealmente, ambas as curvas devem aumentar ao longo do tempo e se estabilizar em valores altos.

- **Gráfico de Perda**: Da mesma forma, haverá duas curvas de perda - uma para o treinamento e outra para a validação. Estas curvas devem diminuir ao longo das épocas, indicando que o modelo está aprendendo e melhorando.

### Possíveis Desafios e Soluções

**1. Overfitting**: Se a acurácia de treinamento for significativamente maior que a acurácia de validação, o modelo pode estar sobreajustando os dados de treinamento. Soluções incluem adicionar mais dados de treinamento, usar regularização ou aumentar o aumento de dados (data augmentation).

**2. Underfitting**: Se ambas as acurácias (treinamento e validação) forem baixas, o modelo pode não estar aprendendo o suficiente. Ajustar a arquitetura do modelo, experimentar diferentes hiperparâmetros ou aumentar a complexidade do modelo pode ajudar.

**3. Balanceamento de Dados**: Certifique-se de que o dataset esteja balanceado para evitar viés no modelo. Um dataset desbalanceado pode resultar em um modelo que performa bem em uma classe, mas mal na outra.

### Próximos Passos

- **Avaliação Adicional**: Utilize conjuntos de dados de teste adicionais que não foram usados durante o treinamento para avaliar a performance real do modelo.

- **Implementação**: Após confirmar a precisão do modelo, você pode implementar o modelo em um ambiente de produção para classificar imagens novas e desconhecidas de cães e gatos.

- **Aprimoramento Contínuo**: Continue coletando novos dados e re-treinando o modelo periodicamente para melhorar a precisão e adaptabilidade a novos padrões.

Esse projeto demonstra uma aplicação prática de Transfer Learning e reflete as boas práticas de aprendizado de máquina, como a validação cruzada e o fine-tuning.

---
