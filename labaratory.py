import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from naive_bayes import train_bayes, predict_bayes
from decision_tree import train_tree, predict_tree
from visualization import draw_boundary, draw_tree
from data_generator import generate_data

from experiments import (
    overfitting_experiment,
    split_three,
    find_best_depth,
    calculate_accuracy,
    compare_algorithms,
    draw_comparison
)


# НАСТРОЙКИ СТРАНИЦЫ

st.set_page_config(
    page_title="AI Learning Lab",
    page_icon="🧠",
    layout="wide"
)


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def clear_training():

    st.session_state.pop("training_result", None)


def clear_results():

    keys = [
        "training_result",
        "overfitting_result",
        "validation_result",
        "comparison",
        "comparison_depth"
    ]

    for key in keys:
        st.session_state.pop(key, None)


def reset_data():

    st.session_state.red_points = [
        [1, 2], [2, 1], [2, 3],
        [3, 2], [3, 4]
    ]

    st.session_state.blue_points = [
        [6, 7], [7, 6], [7, 8],
        [8, 7], [9, 9]
    ]

    clear_results()


def prepare_data():

    X = []
    y = []

    for point in st.session_state.red_points:
        X.append(point)
        y.append(0)

    for point in st.session_state.blue_points:
        X.append(point)
        y.append(1)

    return X, y


def draw_graph():

    fig, ax = plt.subplots(figsize=(6, 5))

    red = st.session_state.red_points
    blue = st.session_state.blue_points

    if red:
        ax.scatter(
            [p[0] for p in red],
            [p[1] for p in red],
            color="red",
            label="Class 0"
        )

    if blue:
        ax.scatter(
            [p[0] for p in blue],
            [p[1] for p in blue],
            color="blue",
            label="Class 1"
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


if "red_points" not in st.session_state:
    reset_data()


# ЗАГОЛОВОК

st.title("🧠 AI Learning Lab")

st.caption(
    "Интерактивная лаборатория машинного обучения"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Данные",
    "🧠 Обучение",
    "🔬 Эксперименты",
    "📚 Теория"
])


# ВКЛАДКА 1. ДАННЫЕ

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Генерация данных")

        count = st.slider(
            "Количество точек",
            20, 500, 100, step=20
        )

        spread = st.slider(
            "Разброс точек",
            0.5, 3.0, 1.0, step=0.1
        )

        noise = st.slider(
            "Уровень шума (%)",
            0, 40, 0, step=5
        )

        if st.button("Сгенерировать данные"):

            red, blue = generate_data(
                count, spread, noise
            )

            st.session_state.red_points = red
            st.session_state.blue_points = blue

            clear_results()

            st.success("Данные сгенерированы")

        with st.expander("Добавить точку вручную"):

            point_class = st.radio(
                "Класс точки",
                ["Красный", "Синий"]
            )

            x = st.slider(
                "Координата X", 0.0, 10.0, 5.0
            )

            y = st.slider(
                "Координата Y", 0.0, 10.0, 5.0
            )

            if st.button("Добавить точку"):

                if point_class == "Красный":
                    st.session_state.red_points.append([x, y])
                else:
                    st.session_state.blue_points.append([x, y])

                clear_results()

        if st.button("Сбросить данные"):
            reset_data()

    with col2:

        st.subheader("Распределение данных")

        draw_graph()

        red_count = len(st.session_state.red_points)
        blue_count = len(st.session_state.blue_points)

        a, b, c = st.columns(3)

        a.metric("🔴 Красные", red_count)
        b.metric("🔵 Синие", blue_count)
        c.metric("Всего", red_count + blue_count)


# ВКЛАДКА 2. ОБУЧЕНИЕ

with tab2:

    st.subheader("Настройки модели")

    algorithm = st.selectbox(
        "Алгоритм машинного обучения",
        ["Наивный Байес", "Дерево решений"],
        on_change=clear_training
    )

    max_depth = 3

    if algorithm == "Дерево решений":

        max_depth = st.slider(
            "Максимальная глубина дерева",
            1, 10, 3,
            on_change=clear_training
        )

    test_percent = st.slider(
        "Процент тестовых данных",
        20, 50, 20, step=10,
        on_change=clear_training
    )

    if st.button("Обучить модель", type="primary"):

        X, y = prepare_data()

        try:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_percent / 100,
                random_state=42,
                stratify=y
            )

            if algorithm == "Наивный Байес":

                model = train_bayes(X_train, y_train)
                predict = predict_bayes

            else:

                model = train_tree(
                    X_train, y_train, max_depth
                )

                predict = predict_tree

            train_accuracy = calculate_accuracy(
                model, predict, X_train, y_train
            )

            test_accuracy = calculate_accuracy(
                model, predict, X_test, y_test
            )

            st.session_state.training_result = {
                "model": model,
                "algorithm": algorithm,
                "predict": predict,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            }

            st.success("Модель успешно обучена!")

        except ValueError as error:

            clear_training()
            st.error(f"Ошибка обучения: {error}")

    if "training_result" in st.session_state:

        result = st.session_state.training_result

        model = result["model"]
        predict = result["predict"]

        st.divider()

        st.subheader("Результаты обучения")

        a, b, c = st.columns(3)

        a.metric(
            "Обучение",
            f'{result["train_accuracy"]:.1f}%'
        )

        b.metric(
            "Тестирование",
            f'{result["test_accuracy"]:.1f}%'
        )

        c.metric(
            "Разрыв точности",
            f'{result["train_accuracy"] - result["test_accuracy"]:.1f} п.п.'
        )

        st.caption(
            f'Обучающих точек: {len(result["X_train"])} | '
            f'Тестовых точек: {len(result["X_test"])}'
        )

        left, right = st.columns(2)

        with left:

            st.subheader("Границы классификации")

            draw_boundary(
                model,
                predict,
                result["X_train"],
                result["y_train"],
                result["X_test"],
                result["y_test"]
            )

        with right:

            st.subheader("Проверка модели")

            new_x = st.slider(
                "X новой точки", 0.0, 10.0, 5.0
            )

            new_y = st.slider(
                "Y новой точки", 0.0, 10.0, 5.0
            )

            prediction = predict(
                model, [new_x, new_y]
            )

            if prediction == 0:
                st.success("🔴 Красный класс (0)")
            else:
                st.success("🔵 Синий класс (1)")

            st.write(
                "Координаты точки:",
                new_x, new_y
            )

            st.caption(
                "Предсказание обновляется автоматически."
            )

        with st.expander("Параметры обученной модели"):
            st.write(model)

        with st.expander("Обучающая выборка"):

            rows = []

            for point, cls in zip(
                result["X_train"], result["y_train"]
            ):

                rows.append({
                    "X": round(point[0], 3),
                    "Y": round(point[1], 3),
                    "Класс": cls
                })

            st.dataframe(rows)

        with st.expander("Результаты тестирования"):

            rows = []

            for point, cls in zip(
                result["X_test"], result["y_test"]
            ):

                prediction = predict(model, point)

                rows.append({
                    "X": round(point[0], 3),
                    "Y": round(point[1], 3),
                    "Правильный класс": cls,
                    "Предсказание": prediction,
                    "Результат": "Верно" if prediction == cls else "Ошибка"
                })

            st.dataframe(rows)

        if algorithm == "Дерево решений":

            with st.expander("Структура дерева решений"):

                st.caption(
                    "Для наглядности рекомендуется глубина до 4."
                )

                if st.checkbox("Показать дерево"):
                    draw_tree(model)

    else:

        st.info("Сначала обучите модель")


# ВКЛАДКА 3. ЭКСПЕРИМЕНТЫ

with tab3:

    st.subheader("Исследование машинного обучения")

    st.caption(
        "Обучение — 60%, валидация — 20%, тестирование — 20%"
    )

    experiment = st.radio(
        "Выберите эксперимент",
        [
            "Переобучение",
            "Подбор глубины",
            "Сравнение алгоритмов"
        ],
        horizontal=True
    )

    X, y = prepare_data()

    if len(X) < 20 or min(y.count(0), y.count(1)) < 5:

        st.warning(
            "Сгенерируйте больше точек обоих классов."
        )

    else:

        # ЭКСПЕРИМЕНТ С ПЕРЕОБУЧЕНИЕМ

        if experiment == "Переобучение":

            st.write(
                "Зависимость точности от глубины дерева."
            )

            if st.button("Запустить эксперимент"):

                X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y)

                depths, train_scores, val_scores = overfitting_experiment(
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )

                st.session_state.overfitting_result = {
                    "depths": depths,
                    "train": train_scores,
                    "validation": val_scores
                }

            if "overfitting_result" in st.session_state:

                result = st.session_state.overfitting_result

                fig, ax = plt.subplots(figsize=(7, 4))

                ax.plot(
                    result["depths"],
                    result["train"],
                    marker="o",
                    label="Training"
                )

                ax.plot(
                    result["depths"],
                    result["validation"],
                    marker="o",
                    label="Validation"
                )

                ax.set_xlabel("Tree depth")
                ax.set_ylabel("Accuracy (%)")

                ax.set_xticks(result["depths"])
                ax.set_ylim(0, 105)

                ax.legend()
                ax.grid(alpha=0.3)

                st.pyplot(fig)
                plt.close(fig)

                rows = []

                for i in range(len(result["depths"])):

                    rows.append({
                        "Глубина": result["depths"][i],
                        "Обучение (%)": round(result["train"][i], 2),
                        "Валидация (%)": round(result["validation"][i], 2),
                        "Разрыв (п.п.)": round(
                            result["train"][i] -
                            result["validation"][i], 2
                        )
                    })

                st.dataframe(rows)

        # АВТОМАТИЧЕСКИЙ ПОДБОР ГЛУБИНЫ

        elif experiment == "Подбор глубины":

            if st.button("Найти оптимальную глубину"):

                X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y)

                best_depth, best_accuracy, results = find_best_depth(
                    X_train, y_train, X_val, y_val
                )

                best_model = train_tree(
                    X_train, y_train, best_depth
                )

                final_accuracy = calculate_accuracy(
                    best_model,
                    predict_tree,
                    X_test,
                    y_test
                )

                st.session_state.validation_result = {
                    "depth": best_depth,
                    "validation": best_accuracy,
                    "testing": final_accuracy,
                    "table": results
                }

            if "validation_result" in st.session_state:

                result = st.session_state.validation_result

                a, b, c = st.columns(3)

                a.metric(
                    "Лучшая глубина",
                    result["depth"]
                )

                b.metric(
                    "Валидация",
                    f'{result["validation"]:.1f}%'
                )

                c.metric(
                    "Итоговый тест",
                    f'{result["testing"]:.1f}%'
                )

                st.dataframe(result["table"])

        # СРАВНЕНИЕ АЛГОРИТМОВ

        else:

            if st.button("Сравнить алгоритмы"):

                results, best_depth = compare_algorithms(X, y)

                st.session_state.comparison = results
                st.session_state.comparison_depth = best_depth

            if "comparison" in st.session_state:

                results = st.session_state.comparison

                st.table(results)

                st.write(
                    "Оптимальная глубина дерева:",
                    st.session_state.comparison_depth
                )

                draw_comparison(results)

                bayes_acc = results[0]["Тестирование"]
                tree_acc = results[1]["Тестирование"]

                if bayes_acc > tree_acc:

                    st.success(
                        "На тестовой выборке лучше справился наивный Байес"
                    )

                elif tree_acc > bayes_acc:

                    st.success(
                        "На тестовой выборке лучше справилось дерево решений"
                    )

                else:

                    st.info(
                        "Алгоритмы показали одинаковую точность"
                    )


# ВКЛАДКА 4. ТЕОРИЯ

with tab4:

    st.header("Основы машинного обучения")

    with st.expander("Что такое машинное обучение?", expanded=True):

        st.write("""
        Машинное обучение — направление искусственного
        интеллекта, в котором алгоритмы находят
        закономерности в данных и используют их
        для предсказания новых результатов.

        В данной лаборатории решается задача
        бинарной классификации.
        """)

    with st.expander("Наивный Байес"):

        st.write("""
        Вероятностный алгоритм классификации,
        основанный на теореме Байеса.
        Предполагает условную независимость признаков.
        """)

        st.latex(
            r"P(C|X)=\frac{P(X|C)P(C)}{P(X)}"
        )

        st.write("Нормальное распределение:")

        st.latex(
            r"f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}"
            r"e^{-\frac{(x-\mu)^2}{2\sigma^2}}"
        )

    with st.expander("Дерево решений"):

        st.write("""
        Дерево решений разделяет данные
        с помощью последовательности условий.

        Для выбора лучшего разделения
        используется критерий Джини.
        """)

        st.latex(r"Gini=1-\sum_{i=1}^{k}p_i^2")

    with st.expander("Обучение, валидация и тестирование"):

        st.markdown("""
        - **Training** — обучение параметров модели.
        - **Validation** — подбор гиперпараметров.
        - **Testing** — итоговая оценка качества модели.
        """)

    with st.expander("Переобучение"):

        st.write("""
        Переобучение возникает, когда модель
        чрезмерно подстраивается под обучающие данные
        и хуже классифицирует новые объекты.

        Ограничение глубины дерева помогает
        уменьшить риск переобучения.
        """)

    with st.expander("Метрика точности"):

        st.latex(
            r"Accuracy=\frac{N_{correct}}{N_{total}}\cdot100\%"
        )

        st.write(
            "Доля правильно классифицированных объектов."
        )


# ПОДВАЛ СТРАНИЦЫ

st.divider()

st.caption(
    "AI Learning Lab | Gaussian Naive Bayes & Decision Tree"
)


        