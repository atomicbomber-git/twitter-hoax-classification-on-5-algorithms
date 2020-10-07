import joblib
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import (
    plot_confusion_matrix,
    precision_recall_fscore_support,
    plot_roc_curve,
    accuracy_score,
    plot_precision_recall_curve,
    confusion_matrix,
)

from constants import *
from train import get_model_file_name, get_test_file_name, get_vectorizer_file_name

pandas.options.display.float_format = '{:.4f}'.format


report_calculation_file = open("./report-extras/report-calculation.txt", "w")


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
    plot_list[key]["roc_ax"] = roc_ax
    plot_list[key]["roc_fig"] = roc_fig

    prc_fig, prc_ax = plt.subplots()
    plot_list[key]["prc_ax"] = prc_ax
    plot_list[key]["prc_fig"] = prc_fig

for fold in range(0, N_FOLDS):
    test_results_per_fold = []

    for algorithm_id in ALGORITHMS:
        model = joblib.load(get_model_file_name(algorithm_id, fold))

        test_file = pandas.read_csv(get_test_file_name(fold))
        data_test = test_file[DATA_KEY]
        target_test = test_file[TARGET_KEY]

        tfidf_vectorizer = joblib.load(get_vectorizer_file_name(fold))

        processed_data_test = tfidf_vectorizer.transform(data_test).toarray()

        predicted_data_test = model.predict(processed_data_test)

        # Plot and save confusion matrix
        plot_confusion_matrix(
            model,
            processed_data_test,
            target_test,
            labels=["f", "h"],
            display_labels=["Fakta", "Hoax"],
            cmap="Greys",
        )

        plt.ylabel("Kelas Prediksi")
        plt.xlabel("Hasil Prediksi")

        plt.savefig(
            "./images/{}_{}_CONFUSION_MATRIX.png".format(
                fold,
                algorithm_id,
            )
        )
        plt.clf()
        plt.close()

        # Plot and save ROC Curve
        name = "{}{}".format(ALGORITHM_SHORT_LABELS[algorithm_id], fold + 1)

        plot_roc_curve(
            model,
            processed_data_test,
            target_test,
            name=name,
            ax=plot_list[algorithm_id]["roc_ax"],
        )

        plot_precision_recall_curve(
            model,
            processed_data_test,
            target_test,
            name=name,
            ax=plot_list[algorithm_id]["prc_ax"],
        )

        tn, fp, fn, tp = confusion_matrix(
            target_test,
            predicted_data_test,
        ).ravel()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        print(
            """Pada diagram dibawah, dapat dilihat bahwa untuk fold ke-{} pada algoritma {}, jumlah true negative (tn) = {:.2f}, false positive (fp) = {:.2f}, false negative (fn) = {:.2f}; Dan true positive (tp) = {:.2f}. Maka nilai precision = tp / (tp + fp) =  {:.2f} / ({:.2f} + {:.2f})= {:.2f}; Nilai recall = tp / (tp + fn) = {:.2f} / ({:.2f} + {:.2f}) = {:.2f}; F1-score = 2 x precision x recall / (precision + recall) = 2 x {:.2f} x {:.2f} / ({:.2f} + {:.2f}) = {:.2f}; Nilai accuracy = tp + tn / (tp + fp + tn + fn) = {:.2f} + {:.2f} / ({:.2f} + {:.2f} + {:.2f} + {:.2f}) = {:.2f}.\n
            """.format(
                fold + 1, 
                ALGORITHM_LABELS[algorithm_id],
                tn, fp, fn, tp,
                tp, tp, fp, precision,
                tp, tp, fn, recall,
                precision, recall, precision, recall, f1_score,
                tp, tn, tp, fp, tn, fn, accuracy,
            ).strip(
            ),
            file=report_calculation_file
        )
        
        precision, recall, f_score, support = precision_recall_fscore_support(
            target_test,
            predicted_data_test,
            pos_label="h",
            zero_division=0,
            average="binary",
        )

        accuracy = accuracy_score(target_test, predicted_data_test)

        test_results_per_algorithm[algorithm_id].append(
            {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f_score,
                "Accuracy": accuracy,
            }
        )

        test_results_per_fold.append(
            {
                "Algoritma": ALGORITHM_LABELS[algorithm_id],
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f_score,
                "Accuracy": accuracy,
            }
        )

    print(
        "Berikut merupakan tabel hasil pengujian untuk fold {}.\n".format(
            fold + 1
        ),
        file=report_calculation_file
    )

    pandas.DataFrame(
        test_results_per_fold,
    ).to_excel(get_test_result_file_name(fold + 1))

for algorithm_id in plot_list:
    roc_fig = plot_list[algorithm_id]["roc_fig"]
    roc_fig.savefig("./images/ROC_CURVE_{}.png".format(algorithm_id))

    prc_fig = plot_list[algorithm_id]["prc_fig"]
    prc_fig.savefig("./images/PPC_CURVE_{}.png".format(algorithm_id))
    pass

averages_list = []

for algorithm_id, test_result in test_results_per_algorithm.items():
    averages_list.append({
        "Algorithm": ALGORITHM_LABELS[algorithm_id],
        **pandas.DataFrame(test_result).mean().to_dict()
    })

report_average_df = pandas.DataFrame(
    averages_list
)
report_average_df.set_index(
    "Algorithm",
    inplace=True
)

print(
    report_average_df.agg({
        'Precision': ['min', 'max'],
        'Recall': ['min', 'max'],
        'F1-Score': ['min', 'max'],
        'Accuracy': ['min', 'max'],
    })
)

report_average_df.style.highlight_max(color='Yellow')
report_average_df.style.highlight_min(color='Red')

report_average_df.to_excel(
    "./test_result_data/Rata-Rata Hasil Penelitian.xlsx"
)