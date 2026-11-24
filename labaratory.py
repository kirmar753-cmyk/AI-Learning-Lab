import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Learning Lab")

st.title("AI Learning Lab")
st.write("Интерактивная лаборатория машинного обучения")

algorithm = st.selectbox(
    "Выберите алгоритм обучения",
    ["Наивный Байес", "Дерево решений"]
)

# Начальные обучающие данные
def reset_data():
    st.session_state.red_points = [
        [1, 2], [2, 1], [2, 3],
        [3, 2], [3, 4]
    ]

    st.session_state.blue_points = [
        [6, 7], [7, 6], [7, 8],
        [8, 7], [9, 9]
    ]

# Создание данных при первом запуске
if "red_points" not in st.session_state:
    reset_data()

# Настройки новой точки
st.subheader("Добавление обучающих данных")

point_class = st.radio(
    "Выберите класс точки",
    ["Красный", "Синий"]
)

x = st.slider("Координата X", 0.0, 10.0, 5.0)
y = st.slider("Координата Y", 0.0, 10.0, 5.0)

# Добавление новой точки
if st.button("Добавить точку"):
    if point_class == "Красный":
        st.session_state.red_points.append([x, y])
    else:
        st.session_state.blue_points.append([x, y])

# Сброс данных
if st.button("Сбросить данные"):
    reset_data()

# Функция построения графика
def draw_graph():
    fig, ax = plt.subplots()

    for point in st.session_state.red_points:
        ax.scatter(point[0], point[1], color="red")

    for point in st.session_state.blue_points:
        ax.scatter(point[0], point[1], color="blue")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

draw_graph()

# Количество точек
red_count = len(st.session_state.red_points)
blue_count = len(st.session_state.blue_points)

st.write("Красных точек:", red_count)
st.write("Синих точек:", blue_count)
st.write("Всего точек:", red_count + blue_count)

if st.button("Обучить ИИ"):
    st.success("Выбран алгоритм: " + algorithm)