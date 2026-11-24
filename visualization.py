
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.colors import ListedColormap


def draw_boundary(model, predict, X_train, y_train,
                  X_test, y_test):

    # Создаём координатную сетку
    x_values = np.linspace(0, 10, 100)
    y_values = np.linspace(0, 10, 100)

    xx, yy = np.meshgrid(x_values, y_values)

    Z = np.zeros(xx.shape)

    # Предсказание класса каждой точки сетки
    for i in range(len(y_values)):
        for j in range(len(x_values)):

            point = [xx[i, j], yy[i, j]]

            Z[i, j] = predict(model, point)

    # Создание графика
    fig, ax = plt.subplots()

    colors = ListedColormap(["red", "blue"])

    ax.contourf(
        xx, yy, Z,
        levels=[-0.5, 0.5, 1.5],
        cmap=colors,
        alpha=0.2
    )

    # Обучающие точки
    for i in range(len(X_train)):

        color = "red" if y_train[i] == 0 else "blue"

        ax.scatter(
            X_train[i][0],
            X_train[i][1],
            color=color,
            marker="o"
        )

    # Тестовые точки
    for i in range(len(X_test)):

        color = "red" if y_test[i] == 0 else "blue"

        ax.scatter(
            X_test[i][0],
            X_test[i][1],
            facecolors="none",
            edgecolors=color,
            marker="s",
            s=80
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Decision Boundary")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

def draw_tree(model):

    fig, ax = plt.subplots(figsize=(12, 6))

    def draw_node(node, x, y, width):

        # Если узел является листом
        if "class" in node:

            cls = node["class"]

            color = "salmon" if cls == 0 else "skyblue"

            ax.text(
                x, y,
                f"Class {cls}",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    facecolor=color
                )
            )

            return

        # Определяем признак
        feature = "X" if node["feature"] == 0 else "Y"

        threshold = node["threshold"]

        ax.text(
            x, y,
            f"{feature} <= {threshold:.2f}",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round",
                facecolor="lightgray"
            )
        )

        left_x = x - width
        right_x = x + width
        next_y = y - 1

        # Левая ветвь
        ax.plot(
            [x, left_x],
            [y, next_y],
            color="gray"
        )

        # Правая ветвь
        ax.plot(
            [x, right_x],
            [y, next_y],
            color="gray"
        )

        ax.text(
            (x + left_x) / 2,
            y - 0.4,
            "Yes",
            fontsize=8
        )

        ax.text(
            (x + right_x) / 2,
            y - 0.4,
            "No",
            fontsize=8
        )

        draw_node(
            node["left"],
            left_x,
            next_y,
            width / 2
        )

        draw_node(
            node["right"],
            right_x,
            next_y,
            width / 2
        )

    draw_node(model, 0, 0, 4)

    ax.set_title("Decision Tree")
    ax.axis("off")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
