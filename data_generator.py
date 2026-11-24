import random


def generate_data(count, spread, noise):

    red_points = []
    blue_points = []

    for i in range(count):

        # Определяем исходный класс
        if i < count // 2:
            center = 3
            cls = 0
        else:
            center = 7
            cls = 1

        # Генерируем координаты
        x = random.gauss(center, spread)
        y = random.gauss(center, spread)

        # Ограничиваем координаты
        x = max(0, min(10, x))
        y = max(0, min(10, y))

        # Добавляем шум
        if random.random() < noise / 100:
            cls = 1 - cls

        # Сохраняем точку
        if cls == 0:
            red_points.append([x, y])
        else:
            blue_points.append([x, y])

    return red_points, blue_points