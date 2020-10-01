import joblib
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support, plot_roc_curve, accuracy_score, \
    plot_precision_recall_curve

from constants import *
from train import get_model_file_name, get_test_file_name, get_vectorizer_file_name


def get_test_result_file_name(fold):
    return "./test_result_data/hasil_uji_{}.xlsx".format(fold)


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

plot_list = {
    NAIVE_BAYES_ID: {},
    SVM_ID: {},
    MULTILAYER_PERCEPTRON_ID: {},
    DECISION_TREE_ID: {},
    RANDOM_FOREST_ID: {},
}

for key in plot_list:
    roc_fig, roc_ax = plt.subplots()
    plot_list[key]['roc_ax'] = roc_ax
    plot_list[key]['roc_fig'] = roc_fig

    prc_fig, prc_ax = plt.subplots()
    plot_list[key]['prc_ax'] = prc_ax
    plot_list[key]['prc_fig'] = prc_fig

for fold in range(0, N_FOLDS):
    test_results_per_fold = []

    for algorithm_id in ALGORITHMS:
        model = joblib.load(
            get_model_file_name(algorithm_id, fold)
        )

        test_file = pandas.read_csv(get_test_file_name(fold))
        data_test = test_file[DATA_KEY]
        target_test = test_file[TARGET_KEY]

        tfidf_vectorizer = joblib.load(
            get_vectorizer_file_name(fold)
        )

        processed_data_test = tfidf_vectorizer.transform(
            data_test
        ).toarray()

        predicted_data_test = model.predict(processed_data_test)

        # Plot and save confusion matrix
        plot_confusion_matrix(
            model,
            processed_data_test,
            target_test,
            labels=['f', 'h'],
            display_labels=['Fakta', 'Hoax'],
            cmap='Greys'
        )
        plt.ylabel('Kelas Prediksi')
        plt.xlabel('Hasil Prediksi')
        plt.savefig("./images/CONFUSION_MATRIX_{}_FOLD_{}.png".format(
            algorithm_id,
            fold,
        ))
        plt.clf()
        plt.close()

        # Plot and save ROC Curve
        name = "{}{}".format(
            ALGORITHM_SHORT_LABELS[algorithm_id],
            fold + 1
        )

        plot_roc_curve(
            model,
            processed_data_test,
            target_test,
            name=name,
            ax=plot_list[algorithm_id]['roc_ax']
        )

        plot_precision_recall_curve(
            model,
            processed_data_test,
            target_test,
            name=name,
            ax=plot_list[algorithm_id]['prc_ax']
        )

        precision, recall, f_score, support = precision_recall_fscore_support(
            target_test,
            predicted_data_test,
            pos_label='h',
            zero_division=0,
            average='binary',
        )

        accuracy = accuracy_score(target_test, predicted_data_test)

        test_results_per_algorithm[algorithm_id].append({
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

        test_results_per_fold.append({
            "Algoritma": ALGORITHM_LABELS[algorithm_id],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f_score,
            "Accuracy": accuracy,
        })

    pandas.DataFrame(
        test_results_per_fold,
    ).to_excel(
        get_test_result_file_name(fold + 1)
    )

for algorithm_id in plot_list:
    roc_fig = plot_list[algorithm_id]['roc_fig']
    roc_fig.savefig(
        "./images/ROC_CURVE_{}.png".format(
            algorithm_id
        )
    )

    prc_fig = plot_list[algorithm_id]['prc_fig']
    prc_fig.savefig(
        "./images/PPC_CURVE_{}.png".format(
            algorithm_id
        )
    )

for algorithm_id, test_result in test_results_per_algorithm.items():
    data_frame = pandas.DataFrame(
        test_result
    )

    mean = data_frame.mean()

    mean.to_excel(
        get_test_result_file_name(
            "Rata-Rata " + ALGORITHM_LABELS[algorithm_id]
        )
    )
