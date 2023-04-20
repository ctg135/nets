# Написать нейросеть, которая различает фигуры 
# 1. Алгоритм определения фигуры (Суммировать значения весов?)
# 2. Организовать датасеты для фигур
#    - После обучения надо сохранять полученный результат, чтобы потом без обучения юзать
# 3. Обучить и проверить нейросеть

# 1. Входные изображения для нейросети размером будут 16х16, черно-белые
# Каждый пиксель это один нейрон, итого 16х16=256 нейронов
# Каждый нейрон имеет свой вес в графе
# В результате суммы графов получается результат фигура или не фигура

import os, numpy as np, json
from PIL import Image

# Функция определения фигуры
def think(weights, values):
    result = 0
    for i in range(len(weights)):
        result += weights[i] * values[i]
    return result

# Расчет погрешности значения
def error(value, desired_value):
    return desired_value - value 

# Функция обучения нейросети
def learn(weights, values, learning_rate):
    for i in range(len(values)):
        weights[i] += values[i] * learning_rate

# Функция обучения нейросети датасетам
def learn_datasets(weights, dataset_directory, learning_rate):
    print('before learning')
    square_print_16x16(weights)
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            values = image_to_values(root[2:] + file)
            learn(weights, values, learning_rate)
    print('\nafter learning')
    square_print_16x16(weights)

# Функция преобразования изображения в массив значений
# TODO Доделать для подгона размера изображения, если оно больше или меньше
def image_to_values(filename):
    obj = Image.open(filename)
    img = obj.load()
    size = obj.size
    values = np.zeros(size[0]*size[1])
    for x in range(size[0]):
        for y in range(size[1]):
            # Если не белый, то заполняем массив 1
            if img[x, y] != (255, 255, 255):
                values[x + y*size[1]] = 1
    return values

# Функция выгрузки параметров в файл
def export_weights(weights, filename):
    file = open(filename, 'w')
    file.write(json.dumps(weights))

# Функция загрузки параметров из файла
def import_weights(filename):
    file = open(filename, 'r')
    return json.load(file)

def square_print_16x16(arr):
    for x in range(16):
        st = ''
        for y in range(16):
            st = st + ' ' + str(arr[x + y*16])
        print(st)

weights = []
for i in range(256):
    weights.append(0)

learn_datasets(weights, '.\\datasets\\square\\', 0.1)

# export_weights(weights, 'settings/square.json')
# weights = import_weights('settings/square.json')



