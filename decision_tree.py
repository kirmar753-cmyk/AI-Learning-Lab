# Индекс Джини
def gini(labels):

    if len(labels) == 0:
        return 0

    p0 = labels.count(0) / len(labels)
    p1 = labels.count(1) / len(labels)

    return 1 - p0 ** 2 - p1 ** 2


# Разделение данных по условию
def split_data(X, y, feature, threshold):

    X_left = []
    y_left = []

    X_right = []
    y_right = []

    for i in range(len(X)):

        if X[i][feature] <= threshold:

            X_left.append(X[i])
            y_left.append(y[i])

        else:

            X_right.append(X[i])
            y_right.append(y[i])

    return X_left, y_left, X_right, y_right


# Поиск лучшего разделения
def best_split(X, y):

    best_gini = gini(y)

    best_feature = None
    best_threshold = None

    for feature in range(2):

        values = sorted(set(
            point[feature] for point in X
        ))

        for i in range(len(values) - 1):

            threshold = (
                values[i] + values[i + 1]
            ) / 2

            X_left, y_left, X_right, y_right = split_data(
                X, y, feature, threshold
            )

            weighted_gini = (
                len(y_left) * gini(y_left)
                + len(y_right) * gini(y_right)
            ) / len(y)

            if weighted_gini < best_gini:

                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# Определение преобладающего класса
def majority_class(y):

    if y.count(0) >= y.count(1):
        return 0
    else:
        return 1


# Обучение дерева решений
def train_tree(X, y, max_depth=3, depth=0):

    # Условия остановки рекурсии
    if gini(y) == 0 or depth >= max_depth:

        return {
            "class": majority_class(y)
        }

    # Поиск лучшего условия
    feature, threshold = best_split(X, y)

    # Если разделение невозможно
    if feature is None:

        return {
            "class": majority_class(y)
        }

    # Разделение данных
    X_left, y_left, X_right, y_right = split_data(
        X, y, feature, threshold
    )

    # Построение узла дерева
    return {
        "feature": feature,
        "threshold": threshold,

        "left": train_tree(
            X_left, y_left,
            max_depth, depth + 1
        ),

        "right": train_tree(
            X_right, y_right,
            max_depth, depth + 1
        )
    }


# Предсказание класса
def predict_tree(model, point):

    # Если достигли листа дерева
    if "class" in model:
        return model["class"]

    feature = model["feature"]
    threshold = model["threshold"]

    if point[feature] <= threshold:

        return predict_tree(
            model["left"], point
        )

    else:

        return predict_tree(
            model["right"], point
        )

if __name__ == "__main__":

    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [6, 7],
        [7, 8],
        [9, 9]
    ]

    y = [0, 0, 0, 1, 1, 1]

    model = train_tree(X, y, max_depth=3)

    print("Обученное дерево:")
    print(model)

    print("Точка [2, 2]:",
          predict_tree(model, [2, 2]))

    print("Точка [8, 8]:",
          predict_tree(model, [8, 8]))