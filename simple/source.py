
# данные для обучения
learn_input = [0, 1, 0, 1]

# Входные данные для проверки работы сети
inputs = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

# Веса графа
weights = [0, 0, 0, 0]
# Получение результата
desired_result = 1
# Коэффициент обучения
learning_rate = 0.1
# Количество циклов для обучения
trials = 5

# Выполнение логики нейросети
def exec_neural_net(data, weights):
    result = 0
    for i in range(len(data)):
        result += data[i] * weights[i]
    return result

# Обучение нейросети
def learn(learn_data, weights, learning_rate):
    for i in range(len(learn_data)):
        weights[i] += learn_data[i] * learning_rate

print('Weights before learning: ', weights)
for i in range(trials):
    learn(learn_input, weights, learning_rate)
print('After learnng: ', weights)

print('\nNeural net work:')
for data in inputs:
    print(data, exec_neural_net(data, weights) == desired_result)



