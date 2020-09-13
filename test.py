import joblib
import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from constants import *
from train import get_model_file_name, get_test_file_name


def get_test_result_file_name(fold):
    return "./test_result_data/hasil_uji_{}.csv".format(fold)

ALGORITHMS = [
    NAIVE_BAYES_ID,
    SVM_ID,
    MULTILAYER_PERCEPTRON_ID,
    DECISION_TREE_ID,
    RANDOM_FOREST_ID,
]

test_results_per_algorithm = {
    NAIVE_BAYES_ID: [],
    SVM_ID: [],
    MULTILAYER_PERCEPTRON_ID: [],
    DECISION_TREE_ID: [],
    RANDOM_FOREST_ID: [],
}

tfidf_vectorizer = joblib.load(TF_IDF_VECTORIZER_NAME)

for fold in range(0, N_FOLDS):

    test_results_per_fold = []

    for algorithm_id in ALGORITHMS:
        model = joblib.load(
            get_model_file_name(algorithm_id, fold)
        )

        test_file = pandas.read_csv(get_test_file_name(fold))
        data_test = test_file[DATA_KEY]
        target_test = test_file[TARGET_KEY]

        processed_data_test = tfidf_vectorizer.transform(
            data_test
        ).toarray()

        predicted_data_test = model.predict(processed_data_test)

        precision, recall, f_score, support = precision_recall_fscore_support(
            target_test,
            predicted_data_test,
        )

        accuracy = accuracy_score(target_test, predicted_data_test)

        test_results_per_algorithm[algorithm_id].append({
            "Presisi": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

        test_results_per_fold.append({
            "Algoritma": ALGORITHM_LABELS[algorithm_id],
            "Presisi": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

    pandas.DataFrame(
        test_results_per_fold,
    ).to_csv(
        get_test_result_file_name(fold + 1)
    )

for algorithm_id, test_result in test_results_per_algorithm.items():
    data_frame = pandas.DataFrame(
        test_result
    )

    mean = data_frame.mean()

    mean.to_csv(
        get_test_result_file_name(
            "Rata-Rata " + ALGORITHM_LABELS[algorithm_id]
        )
    )
