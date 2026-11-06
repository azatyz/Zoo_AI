import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


class BaseExtractionModel:
    def __init__(self, name):
        self.name = name

    def predict(self, stages, feed_conc, flow_ratio):
        # Метод переопределяется в наследниках
        pass

class ClassicModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("Классическая (Эмпирическая)")

    def predict(self, stages, feed_conc, flow_ratio):
        # Имитация математической модели: экспоненциальное снижение концентрации
        # Чем больше отношение потоков (flow_ratio), тем лучше идет экстракция
        efficiency = 0.15 * flow_ratio
        concentrations = [feed_conc]
        
        for i in range(1, stages + 1):
            next_conc = concentrations[-1] * (1 - efficiency)
            concentrations.append(max(next_conc, 0.01)) # Не уходим в минус
            
        recovery = (feed_conc - concentrations[-1]) / feed_conc * 100
        return concentrations, recovery

class MLModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("ML-модель (Random Forest - Заглушка)")

    def predict(self, stages, feed_conc, flow_ratio):
        # Имитация ML: берем классику и накидываем "шум" предсказаний
        base_model = ClassicModel()
        base_conc, _ = base_model.predict(stages, feed_conc, flow_ratio)
        
        # Добавляем случайные отклонения, имитируя работу ML
        noise = np.random.normal(0, feed_conc * 0.02, stages + 1)
        ml_conc = np.clip(base_conc + noise, 0.01, feed_conc)
        ml_conc[0] = feed_conc # Входная концентрация всегда точная
        
        recovery = (feed_conc - ml_conc[-1]) / feed_conc * 100
        return ml_conc, recovery

class HybridModel(BaseExtractionModel):
    def __init__(self):
        super().__init__("Гибридная (Блендинг Classic + ML)")

    def predict(self, stages, feed_conc, flow_ratio):
        # Гибрид: берем среднее между классикой и ML
        classic_conc, _ = ClassicModel().predict(stages, feed_conc, flow_ratio)
        ml_conc, _ = MLModel().predict(stages, feed_conc, flow_ratio)
        
        hybrid_conc = (np.array(classic_conc) + np.array(ml_conc)) / 2
        recovery = (feed_conc - hybrid_conc[-1]) / feed_conc * 100
        return hybrid_conc, recovery

# --- Интерфейс Streamlit ---

st.set_page_config(page_title="Зоопарк ИИ: Гидрометаллургия", layout="wide")

st.title("🧪 MVP: Зоопарк ИИ-моделей для экстракционных каскадов")
st.markdown("Заказчик: ООО «ЦИФРОВЫЕ РЕШЕНИЯ» | Проектная практика")

# Боковая панель с параметрами
st.sidebar.header("Входные параметры")
stages = st.sidebar.slider("Число ступеней каскада", min_value=1, max_value=20, value=8)
feed_conc = st.sidebar.number_input("Концентрация металла на входе (г/л)", value=100.0)
flow_ratio = st.sidebar.slider("Отношение потоков (O:A)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

# Выбор моделей
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

# Окно результатов
if st.sidebar.button("Запустить симуляцию", type="primary"):
    if not selected_models:
        st.warning("Выберите хотя бы одну модель для расчета!")
    else:
        st.subheader("Результаты моделирования")
        
        # Настройка графика
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Номер ступени каскада")
        ax.set_ylabel("Концентрация металла (г/л)")
        ax.set_title("Профиль концентрации по ступеням")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(range(stages + 1))
        
        cols = st.columns(len(selected_models))
        
        # Расчет и отрисовка для каждой выбранной модели
        for idx, model_name in enumerate(selected_models):
            model = available_models[model_name]
            concentrations, recovery = model.predict(stages, feed_conc, flow_ratio)
            
            # Строим график
            ax.plot(range(stages + 1), concentrations, marker='o', label=model.name, linewidth=2)
            
            # Выводим метрики в колонки
            with cols[idx]:
                st.info(f"**{model.name}**\n\nСтепень извлечения: **{recovery:.2f}%**\n\nОстаток: **{concentrations[-1]:.2f} г/л**")
                
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Настройте параметры в панели слева и нажмите «Запустить симуляцию».")
