import math


# Среднее арифметическое
def mean(values):
    return sum(values) / len(values)


# Дисперсия
def variance(values):
    avg = mean(values)

    result = 0

    for value in values:
        result += (value - avg) ** 2

    return result / len(values) + 0.000001


# Обучение модели
def train_bayes(X, y):
    model = {}

    for cls in [0, 1]:

        x_values = []
        y_values = []

        for i in range(len(X)):
            if y[i] == cls:
                x_values.append(X[i][0])
                y_values.append(X[i][1])

        model[cls] = {
            "mean_x": mean(x_values),
            "mean_y": mean(y_values),
            "var_x": variance(x_values),
            "var_y": variance(y_values),
            "prior": len(x_values) / len(X)
        }

    return model


# Логарифм плотности вероятности
def gaussian(value, avg, var):
    return (
        -0.5 * math.log(2 * math.pi * var)
        - (value - avg) ** 2 / (2 * var)
    )


# Предсказание класса точки
def predict_bayes(model, point):
    scores = []

    for cls in [0, 1]:

        data = model[cls]

        score = math.log(data["prior"])

        score += gaussian(
            point[0],
            data["mean_x"],
            data["var_x"]
        )

        score += gaussian(
            point[1],
            data["mean_y"],
            data["var_y"]
        )

        scores.append(score)

    if scores[0] >= scores[1]:
        return 0
    else:
        return 1
