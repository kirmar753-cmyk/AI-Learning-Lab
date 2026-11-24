import matplotlib.pyplot as plt
import streamlit as st

from decision_tree import train_tree, predict_tree
from sklearn.model_selection import train_test_split
from naive_bayes import train_bayes, predict_bayes


# Расчёт точности модели
def calculate_accuracy(model, predict, X, y):

    correct = 0

    for i in range(len(X)):

        prediction = predict(model, X[i])

        if prediction == y[i]:
            correct += 1

    return correct / len(X) * 100


# Эксперимент с глубиной дерева
def overfitting_experiment(
    X_train, y_train, X_test, y_test
):

    depths = []
    train_scores = []
    test_scores = []

    for depth in range(1, 11):

        model = train_tree(
            X_train, y_train, depth
        )

        train_accuracy = calculate_accuracy(
            model, predict_tree, X_train, y_train
        )

        test_accuracy = calculate_accuracy(
            model, predict_tree, X_test, y_test
        )

        depths.append(depth)
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

    return depths, train_scores, test_scores


# Построение графика точности
def draw_overfitting(
    depths, train_scores, test_scores
):

    fig, ax = plt.subplots()

    ax.plot(
        depths,
        train_scores,
        marker="o",
        color="blue",
        label="Training"
    )

    ax.plot(
        depths,
        test_scores,
        marker="o",
        color="green",
        label="Testing"
    )

    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Accuracy (%)")

    ax.set_title("Overfitting experiment")

    ax.set_xticks(depths)
    ax.set_ylim(0, 105)

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)
    # Разделение на три выборки
def split_three(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Поиск оптимальной глубины дерева
def find_best_depth(X_train, y_train, X_val, y_val):

    best_depth = 1
    best_accuracy = -1

    results = []

    for depth in range(1, 11):

        model = train_tree(X_train, y_train, depth)

        accuracy = calculate_accuracy(
            model, predict_tree, X_val, y_val
        )

        results.append({
            "Глубина": depth,
            "Точность (%)": round(accuracy, 2)
        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_depth = depth

    return best_depth, best_accuracy, results

def compare_algorithms(X, y):

    X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y)

    # Наивный Байес
    bayes_model = train_bayes(X_train, y_train)

    # Подбор глубины дерева
    best_depth, _, _ = find_best_depth(
        X_train, y_train, X_val, y_val
    )

    tree_model = train_tree(
        X_train, y_train, best_depth
    )

    results = []

    models = [
        ("Наивный Байес", bayes_model, predict_bayes),
        ("Дерево решений", tree_model, predict_tree)
    ]

    for name, model, predict in models:

        train_acc = calculate_accuracy(
            model, predict, X_train, y_train
        )

        val_acc = calculate_accuracy(
            model, predict, X_val, y_val
        )

        test_acc = calculate_accuracy(
            model, predict, X_test, y_test
        )

        results.append({
            "Алгоритм": name,
            "Обучение": round(train_acc, 2),
            "Валидация": round(val_acc, 2),
            "Тестирование": round(test_acc, 2)
        })

    return results, best_depth

def draw_comparison(results):

    names = ["Training", "Validation", "Testing"]

    bayes_scores = [
        results[0]["Обучение"],
        results[0]["Валидация"],
        results[0]["Тестирование"]
    ]

    tree_scores = [
        results[1]["Обучение"],
        results[1]["Валидация"],
        results[1]["Тестирование"]
    ]

    fig, ax = plt.subplots()

    x = [0, 1, 2]
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        bayes_scores,
        width,
        label="Naive Bayes",
        color="royalblue"
    )

    ax.bar(
        [i + width / 2 for i in x],
        tree_scores,
        width,
        label="Decision Tree",
        color="orange"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Algorithm comparison")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)
