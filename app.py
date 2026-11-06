
import streamlit as st
import matplotlib.pyplot as plt

# Импортируем модели из папки models
from models.classic_model import ClassicModel
from models.ml_model import MLModel
from models.hybrid_model import HybridModel


# Настройки страницы
st.set_page_config(
    page_title="Зоопарк ИИ: Гидрометаллургия",
    layout="wide"
)

# Заголовок приложения
st.title("🧪 Зоопарк ИИ-моделей для экстракционных каскадов")

st.markdown(
    "Заказчик: ООО «ЦИФРОВЫЕ РЕШЕНИЯ» | Проектная практика"
)

st.divider()

# Описание проекта
st.subheader("О проекте")

st.markdown("""
Данное приложение предназначено для моделирования и сравнения
подходов к расчёту экстракционных каскадов в гидрометаллургии.

Пользователь может задавать технологические параметры,
выбирать модели и анализировать изменение концентрации металла
по ступеням каскада.
""")

st.divider()

# Описание доступных моделей
st.subheader("Доступные модели")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Классическая

    Демонстрационная модель, использующая упрощённую
    математическую зависимость для расчёта изменения
    концентрации металла по ступеням.
    """)

with col2:
    st.markdown("""
    ### ML-модель

    Демонстрационная реализация, имитирующая отклонения
    результатов базовой модели с помощью случайного шума.
    """)

with col3:
    st.markdown("""
    ### Гибридная

    Демонстрационная комбинация результатов классической
    и ML-модели путём усреднения.
    """)

st.divider()

st.sidebar.header("Входные параметры")

stages = st.sidebar.slider(
    "Число ступеней каскада",
    min_value=1,
    max_value=20,
    value=8,
    help="Количество ступеней экстракционного каскада."
)

feed_conc = st.sidebar.number_input(
    "Концентрация металла на входе (г/л)",
    min_value=0.01,
    value=100.0,
    help="Начальная концентрация металла в исходном растворе."
)

flow_ratio = st.sidebar.slider(
    "Отношение потоков (O:A)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Отношение органического потока к водному (O:A)."
)

st.sidebar.header("Настройки зоопарка")

available_models = {
    "Классическая": ClassicModel(),
    "ML-модель": MLModel(),
    "Гибридная": HybridModel()
}

selected_models = st.sidebar.multiselect(
    "Выберите модели для сравнения",
    options=list(available_models.keys()),
    default=["Классическая", "Гибридная"]
)

if st.sidebar.button("Запустить симуляцию", type="primary"):

    if not selected_models:
        st.warning("Выберите хотя бы одну модель!")

    else:
        st.subheader("Результаты моделирования")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.set_xlabel("Номер ступени каскада")
        ax.set_ylabel("Концентрация металла (г/л)")
        ax.set_title("Изменение концентрации по ступеням")

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xticks(range(stages + 1))

        columns = st.columns(len(selected_models))
        display_names = {
            "Классическая": "Классическая модель",
            "ML-модель": "ML-модель",
            "Гибридная": "Гибридная модель"
    }

        for index, model_name in enumerate(selected_models):

            model = available_models[model_name]

            concentrations, recovery = model.predict(
                stages,
                feed_conc,
                flow_ratio
            )

            # Добавляем результат модели на график
            ax.plot(
                range(stages + 1),
                concentrations,
                marker="o",
                label=display_names[model_name],
                linewidth=2
            )

            # Вывод численных результатов
            with columns[index]:
                st.markdown(f"### {model.name}")
                display_names = {
                "Классическая": "Классическая модель",
                "ML-модель": "ML-модель",
                "Гибридная": "Гибридная модель"
            }

                st.markdown(f"### {display_names[model_name]}")

                st.metric(
                    "Степень извлечения",
                    f"{recovery:.2f}%"
                )

                st.metric(
                    "Начальная концентрация",
                    f"{feed_conc:.2f} г/л"
                )

                st.metric(
                    "Остаточная концентрация",
                    f"{concentrations[-1]:.2f} г/л",
                    delta=f"{concentrations[-1] - feed_conc:.2f} г/л",
                    delta_color="inverse"
                )

        ax.legend()
        st.pyplot(fig)

        plt.close(fig)

else:
    st.info(
        "Настройте параметры в панели слева "
        "и нажмите «Запустить симуляцию»."
    )
