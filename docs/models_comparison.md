# Сравнение моделей проекта «Зоопарк ИИ»

Обновлено: ноябрь 2026. Все MAE проверены по первоисточникам и скриншотам бенчмарков.

| Модель | Тип архитектуры | Входные данные | Заявленная точность (MAE, эВ) | Железо | Статус в Зоопарке ИИ | Источник |
|---|---|---|---|---|---|---|
| **MXene Gas Sensor ML Models** (Iben Nassar et al.) | Табличные ансамбли scikit-learn / XGBoost: RFR, XGB, GBR, DTR, ANN | Табличные дескрипторы MXene + дескрипторы адсорбата (RDKit/matminer); 600 точек DFT-PBE | RFR: тест MAE **0.32**, RMSE 0.45, R²=0.95; CV MAE 0.41±0.10 | **CPU** (pandas/scikit-learn, инференс — миллисекунды) | Локально на CPU | 2025, J. Phys. Chem. C 129(5), DOI: [10.1021/acs.jpcc.4c08353](https://doi.org/10.1021/acs.jpcc.4c08353); Данные: Zenodo [10.5281/zenodo.22957732](https://zenodo.org/records/22957732) |
| **MACE-MP-0 (small)** | Универсальный межатомный потенциал: эквивариантная GNN (MACE), обучен на MPtrj | 3D-структура (ASE Atoms): slab + адсорбат; E_ads через релаксацию | **~0.36 эВ** (CatBench, поверхности металлов, normal); **0.181 эВ** (MOFs); **0.250 эВ** (оксиды). Без fine-tuning на катализ точность ограничена | **CPU** (ASE MACECalculator, device='cpu', float32) | Локально на CPU | 2023/2025, arXiv:[2401.00096](https://arxiv.org/abs/2401.00096); Код: [github.com/ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **AdsMT** | Мультимодальный трансформер: графовый энкодер (AdsGT/GNN) + MLP адсорбата + cross-attention | Периодический граф поверхности + 208 RDKit-дескрипторов; предсказывает GMAE | Тест MAE: **0.095** (FG-GMAE), **0.143** (Alloy-GMAE), **0.389** (OCD-GMAE, с transfer learning). Инференс на CPU: **десятки мс** для легких энкодеров (AdsGT/CGCNN); тяжелые (GemNet-OC) без GPU не бенчмарчились | Обучение: NVIDIA A100; CPU-инференс возможен для легких энкодеров | Каталог-советчик (GPU*) | 2025, Nature Communications 16, 3232, DOI: [10.1038/s41467-025-58499-7](https://doi.org/10.1038/s41467-025-58499-7); Код: [github.com/schwallergroup/AdsMT](https://github.com/schwallergroup/AdsMT) |
| **LEI-framework (LERN)** | Дескрипторы локального окружения (Вороной) + RF или ResNet-34 (LERN) | Матрица 6×11 (6 дескрипторов × 3 слоя соседей); фокус — H-адсорбция (HER) | RF: MAE **0.14** (H на 3D); LERN (ResNet): **наименьший MAE** среди CGCNN/SchNet/ALIGNN на 3D-материалах (Рис. 4 статьи); устойчивость к выбросам выше | RF — CPU; LERN обучался на GPU (~26 мин на RTX 3080) | Каталог-советчик (GPU)* | 2024, Journal of Materials Informatics 4:4, DOI: [10.20517/jmi.2023.41](https://doi.org/10.20517/jmi.2023.41); Код: [github.com/mpeshel/LEI-framework_LERN](https://github.com/mpeshel/LEI-framework_LERN) |
| **OC20 / FAIRChem (OCP)** | Предобученные GNN: SchNet, DimeNet++, GemNet-OC, eSCN и др. | 3D-граф «slab + адсорбат»; задачи S2EF и IS2RE | Базлайны OC20 IS2RE-direct, тест ID: SchNet **0.639**, DimeNet++ **0.562**; Relaxation-based DimeNet++: **0.512** | NVIDIA GPU + CUDA (чекпойнты рассчитаны на GPU) | Каталог-советчик (GPU) | 2021, ACS Catalysis 11, 6059–6072, DOI: [10.1021/acscatal.0c04525](https://doi.org/10.1021/acscatal.0c04525); Код: [github.com/Open-Catalyst-Project/ocp](https://github.com/Open-Catalyst-Project/ocp) |

\* AdsMT и LEI оставлены в каталоге, т.к. валидированный пайплайн авторов требует GPU (A100/RTX 3080). 
Для AdsMT CPU-инференс теоретически возможен только для легких энкодеров (AdsGT), но не тестировался авторами.
Для LEI RF-ветка потенциально переносима на CPU в следующей итерации.

## Вывод: почему локальный MVP = MXene + MACE-MP-0

1. **MXene (RFR/XGB)** — единственный вариант с гарантированным CPU-инференсом за миллисекунды и MAE ~0.32 эВ на своей предметной области.
2. **MACE-MP-0 small** — единственный универсальный потенциал, работающий на CPU через ASE. MAE ~0.36 эВ на поверхностях металлов (CatBench) приемлем для качественного скрининга, но требует честного предупреждения пользователя.
3. **AdsMT** — лучший MAE (0.095–0.143) среди всех моделей, но привязан к GPU (A100). На CPU легкие энкодеры дадут «десятки мс», но тяжелые (GemNet-OC) непригодны. Оставляем в советчике.
4. **LEI-framework** — специализирован на HER (водород). LERN превосходит аналоги по MAE и устойчивости, но требует сборки пайплайна дескрипторов. RF-ветка (MAE 0.14) — кандидат на локальный перенос.
5. **OC20/FAIRChem** — эталонные GNN, но MAE базлайнов 0.5–0.6 эВ и требование GPU делают их непригодными для локального MVP.
