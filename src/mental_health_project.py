"""
Анализ факторов психического здоровья на основе датасета
"Mental Health in Tech Survey" (Kaggle).

Запуск из корня проекта:
    python src/mental_health_project.py

Перед запуском:
1. Скачайте датасет:
   https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
2. Поместите файл survey.csv в папку data/, чтобы путь был: data/survey.csv
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    classification_report, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


pd.set_option("display.max_columns", 50)
sns.set(style="whitegrid")

DATA_PATH = os.path.join("data", "survey.csv")
RANDOM_STATE = 42
PLOTS_DIR = "plots"


def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета из указанного пути."""
    if not os.path.exists(path):
        print(f"[ОШИБКА] Файл {path} не найден.")
        print("Скачайте 'survey.csv' с Kaggle и поместите в папку data/.")
        sys.exit(1)
    df = pd.read_csv(path)
    print("[OK] Данные загружены.")
    print("Форма датафрейма:", df.shape)
    return df


def clean_gender(g: str) -> str:
    """Грубая нормализация пола (male / female / other)."""
    g = str(g).strip().lower()
    if "female" in g or "woman" in g or g in ["f", "femake"]:
        return "female"
    if "male" in g or "man" in g or g in ["m", "male-ish"]:
        return "male"
    return "other"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка и обработка пропусков."""
    print("\n=== Предобработка данных ===")

    before = df.shape[0]
    df = df[(df["Age"] >= 18) & (df["Age"] <= 65)].copy()
    after = df.shape[0]
    print(f"Фильтрация по возрасту: было {before}, осталось {after}")

    if "Gender" in df.columns:
        df["Gender_clean"] = df["Gender"].apply(clean_gender)
    else:
        print("[ПРЕДУПРЕЖДЕНИЕ] Колонка 'Gender' не найдена, создаём 'Gender_clean=other'")
        df["Gender_clean"] = "other"

    if "treatment" not in df.columns:
        print("[ОШИБКА] В данных нет колонки 'treatment'.")
        sys.exit(1)

    mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1}
    df["treatment"] = df["treatment"].map(mapping)

    if df["treatment"].isna().any():
        most_common = df["treatment"].mode()[0]
        df["treatment"] = df["treatment"].fillna(most_common)

    df["treatment"] = df["treatment"].astype(int)

    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    df[cat_cols] = df[cat_cols].fillna("Unknown")

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Доля пропусков после заполнения:")
    print(df.isna().mean().sort_values(ascending=False).head(10))

    return df


def basic_eda(df: pd.DataFrame) -> None:
    """Простейший EDA: несколько описательных выводов и графиков."""
    print("\n=== Разведочный анализ данных (EDA) ===")
    print("\nПервые 5 строк:")
    print(df.head())

    print("\nИнформация о данных:")
    print(df.info())

    print("\nОписание числовых признаков:")
    print(df.describe().T)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if "treatment" in df.columns:
        plt.figure(figsize=(5, 4))
        sns.countplot(x="treatment", data=df)
        plt.title("Распределение целевой переменной (treatment)")
        plt.xlabel("Treatment (0 = нет, 1 = да)")
        plt.ylabel("Количество")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "treatment_distribution.png")
        plt.savefig(path)
        plt.close()
        print(f"[ГРАФИК] {path} сохранён.")

    if "treatment" in df.columns and "Age" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x="Age", hue="treatment", kde=True, bins=30, stat="density")
        plt.title("Распределение возраста по группам treatment")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "age_by_treatment.png")
        plt.savefig(path)
        plt.close()
        print(f"[ГРАФИК] {path} сохранён.")

        plt.figure(figsize=(5, 4))
        sns.boxplot(x="treatment", y="Age", data=df)
        plt.title("Возраст по группам treatment")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "age_boxplot_by_treatment.png")
        plt.savefig(path)
        plt.close()
        print(f"[ГРАФИК] {path} сохранён.")

    if "family_history" in df.columns and "treatment" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="family_history", hue="treatment", data=df)
        plt.title("Семейная история псих. заболеваний vs treatment")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "family_history_vs_treatment.png")
        plt.savefig(path)
        plt.close()
        print(f"[ГРАФИК] {path} сохранён.")


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Выбор признаков и разделение на X, y."""
    print("\n=== Подготовка признаков ===")

    if "treatment" not in df.columns:
        print("[ОШИБКА] В данных нет колонки 'treatment'.")
        sys.exit(1)

    features = [
        "Age",
        "Gender_clean",
        "family_history",
        "benefits",
        "care_options",
        "work_interfere",
        "no_employees",
        "remote_work",
        "tech_company",
    ]

    missing = [f for f in features if f not in df.columns]
    if missing:
        print("[ПРЕДУПРЕЖДЕНИЕ] Отсутствующие признаки:", missing)
        features = [f for f in features if f in df.columns]

    X = df[features].copy()
    y = df["treatment"].copy()

    cat_features = [c for c in features if X[c].dtype == "object"]
    num_features = [c for c in features if X[c].dtype != "object"]

    print("Используемые признаки:", features)
    print("Категориальные:", cat_features)
    print("Числовые:", num_features)
    print("Уникальные значения y:", sorted(y.unique()))

    return X, y, num_features, cat_features


def build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    """Создаёт препроцессор для числовых и категориальных признаков."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Обучает 3 модели и возвращает таблицу метрик и пайплайны."""

    print("\n=== Обучение моделей ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    results = []
    pipelines: Dict[str, Pipeline] = {}

    for name, clf in models.items():
        print(f"\n--- Модель: {name} ---")
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        print(f"Accuracy: {acc:.3f}")
        print(f"Recall:   {rec:.3f}")
        print(f"F1-score: {f1:.3f}")
        if not np.isnan(auc):
            print(f"AUC ROC:  {auc:.3f}")
        print("\nОтчёт классификации:")
        print(classification_report(y_test, y_pred))

        results.append({"model": name, "accuracy": acc, "recall": rec, "f1": f1, "auc": auc})
        pipelines[name] = pipe

    results_df = pd.DataFrame(results)
    print("\n=== Сводная таблица метрик ===")
    print(results_df)

    return results_df, pipelines, (X_train, X_test, y_train, y_test)


def plot_roc_curves(pipelines: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Строит ROC-кривые для всех моделей с predict_proba."""
    print("\n=== ROC-кривые ===")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(7, 6))

    has_curve = False

    for name, pipe in pipelines.items():
        model = pipe.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            continue
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name)
        has_curve = True

    if not has_curve:
        print("Нет моделей с predict_proba, ROC-кривые не построены.")
        return

    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC-кривые моделей")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"[ГРАФИК] {path} сохранён.")


def plot_feature_importance_tree(
    pipeline: Pipeline,
    num_features: List[str],
    cat_features: List[str]
) -> None:
    """Строит диаграмму важности признаков для DecisionTree."""
    print("\n=== Важность признаков (Decision Tree) ===")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]

    if not isinstance(model, DecisionTreeClassifier):
        print("[ПРЕДУПРЕЖДЕНИЕ] В функцию передана не DecisionTreeClassifier.")
        return

    cat_transformer = pre.named_transformers_.get("cat", None)
    if cat_transformer is None:
        print("[ПРЕДУПРЕЖДЕНИЕ] Не найден трансформер 'cat'.")
        return

    try:
        ohe_features = cat_transformer.get_feature_names_out(cat_features)
    except Exception as e:
        print("[ОШИБКА] Не удалось получить имена OHE-признаков:", e)
        return

    all_features = np.concatenate([np.array(num_features), ohe_features])
    importances = model.feature_importances_

    feat_imp = pd.DataFrame({"feature": all_features, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    print("Топ-15 признаков по важности:")
    print(feat_imp)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp, x="importance", y="feature")
    plt.title("Важность признаков (Decision Tree)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_decision_tree.png")
    plt.savefig(path)
    plt.close()
    print(f"[ГРАФИК] {path} сохранён.")


def main() -> None:
    print("=== Анализ факторов психического здоровья (Mental Health in Tech Survey) ===")

    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    basic_eda(df)

    X, y, num_features, cat_features = prepare_features(df)
    preprocessor = build_preprocessor(num_features, cat_features)

    results_df, pipelines, (X_train, X_test, y_train, y_test) = train_and_evaluate_models(
        X, y, preprocessor
    )

    metrics_path = "model_metrics.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Метрики моделей сохранены в {metrics_path}")

    plot_roc_curves(pipelines, X_test, y_test)

    if "DecisionTree" in pipelines:
        plot_feature_importance_tree(pipelines["DecisionTree"], num_features, cat_features)

    print("\n=== Работа завершена ===")
    print("Основные результаты:")
    print("- Таблица метрик: model_metrics.csv")
    print(f"- Графики сохранены в папке: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
