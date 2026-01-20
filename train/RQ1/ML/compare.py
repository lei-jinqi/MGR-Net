from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

featurepath = "../../../dataset/features/"
labelpath = "../../../dataset/label_6188.csv"


def loadfeature(filename):
    return pd.read_csv(featurepath + filename, index_col="address")


def get_full_data(featurename):
    print(f"\nLoading features from: {featurename} ...")
    try:
        labelset = pd.read_csv(labelpath, index_col="address")
        featuredata1 = loadfeature(featurename)

        datatable = featuredata1.join(labelset)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        if 'label' not in datatable.columns:
            return None, None

        labels = datatable['label']
        features = datatable.drop(columns=['label'])
        features_imputed = imputer.fit_transform(features)

        datatable_imputed = pd.DataFrame(features_imputed, index=features.index, columns=features.columns)
        datatable_imputed = datatable_imputed.join(labels)
        datatable_imputed.dropna(subset=['label'], inplace=True)

        target = datatable_imputed["label"].to_numpy()
        data = datatable_imputed.drop(columns=['label']).to_numpy()
        data = data.astype(np.float32)

        print(f"Data shape: {data.shape}, Label distribution: {Counter(target)}")
        return data, target
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def evaluate(model, data, target, use_smote=False, model_name=None):
    if data is None or target is None:
        return

    pipeline = model
    if use_smote:
        pipeline = Pipeline([
            ('smote', SVMSMOTE(random_state=42)),
            ('model', model)
        ])
        if model_name is None:
            model_name = type(model).__name__ + " (SMOTE)"
    else:
        if model_name is None:
            model_name = type(model).__name__

    print(f"--- Evaluating {model_name} ---")

    scoring = {
        'Precision': 'precision',
        'Recall': 'recall',
        'F1': 'f1',
        'ROC-AUC': 'roc_auc',
        'AUC-PR': 'average_precision'
    }

    try:
        scores = cross_validate(pipeline, data, target, cv=5, scoring=scoring, n_jobs=-1)

        for metric_display_name, metric_key in scoring.items():
            result_key = f"test_{metric_display_name}"
            if result_key in scores:
                metric_scores = scores[result_key]
                print(f"'{model_name}_{metric_display_name}': {list(np.round(metric_scores, 5))},")
                print(f"{metric_display_name} Mean: {metric_scores.mean():.4f} (+/- {metric_scores.std():.4f})")

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
    print("-" * 40)


def main():
    data_1g, target_1g = get_full_data("rawcfg_1gram_tf.csv")

    if data_1g is not None:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
        evaluate(rf, data_1g, target_1g, model_name="RandomForest")


        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        evaluate(xgb, data_1g, target_1g, model_name="XGBoost")

        lgbm = LGBMClassifier(
            n_estimators=100,
            num_leaves=20,
            max_depth=5,
            random_state=42
        )
        evaluate(lgbm, data_1g, target_1g, model_name="LightGBM")

    data_2g, target_2g = get_full_data("rawcfg_2gram_tfidf.csv")

    if data_2g is not None:
        cat = CatBoostClassifier(
            iterations=200,
            depth=5,
            verbose=0,
            random_state=42
        )
        evaluate(cat, data_2g, target_2g, model_name="CatBoost")

    data_3g, target_3g = get_full_data("filteredcfg_3gram_tfidf.csv")

    if data_3g is not None:
        rf_seq = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
        evaluate(rf_seq, data_3g, target_3g, use_smote=True, model_name="Sequence")


if __name__ == "__main__":
    main()