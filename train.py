import joblib
import pandas
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from constants import *

MODEL_COMPRESS_LEVEL = 5


def get_test_file_name(fold):
    return "./test_data/test_{}.csv".format(fold)


def get_model_file_name(algorithm_id, fold):
    return "./model_data/{}_{}.model".format(algorithm_id, fold)


def naive_bayes(data, target, fold):
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(data, target)
    joblib.dump(multinomial_nb, get_model_file_name(NAIVE_BAYES_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def support_vector_machine(data, target, fold):
    support_vector_machine = svm.SVC()
    support_vector_machine.fit(data, target)
    joblib.dump(support_vector_machine, get_model_file_name(SVM_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def perceptron(data, target, fold):
    perceptron = Perceptron()
    perceptron.fit(data, target)
    joblib.dump(perceptron, get_model_file_name(MULTILAYER_PERCEPTRON_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def decision_tree(data, target, fold):
    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(data, target)
    joblib.dump(decision_tree_classifier, get_model_file_name(DECISION_TREE_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def random_forest(data, target, fold):
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(data, target)
    joblib.dump(classifier, get_model_file_name(RANDOM_FOREST_ID, fold), compress=MODEL_COMPRESS_LEVEL)
    pass


def get_vectorizer_file_name(fold):
    return "./vectorizer_data/vectorizer_{}.model".format(fold)


algorithms = {
    NAIVE_BAYES_ID: {"train": naive_bayes},
    SVM_ID: {"train": support_vector_machine},
    MULTILAYER_PERCEPTRON_ID: {"train": perceptron},
    DECISION_TREE_ID: {"train": decision_tree},
    RANDOM_FOREST_ID: {"train": random_forest},
}

if __name__ == "__main__":
    data_frame = pandas.read_csv(
        PREPROCESSED_DATA_FILE,
    )

    data = data_frame[DATA_KEY].to_numpy()
    target = data_frame[TARGET_KEY].to_numpy()

    kFolder = KFold(n_splits=N_FOLDS)
    fold_count = 0

    for train_index, test_index in kFolder.split(data):
        print("Memproses Tweet ke {} - {}".format(
            test_index[0] + 1,
            test_index[-1] + 1
        ))

        data_train, target_train = data[train_index], target[train_index]

        tfidf_vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.7,
        )

        processed_data_train = tfidf_vectorizer.fit_transform(
            data_train
        ).toarray()

        joblib.dump(tfidf_vectorizer, get_vectorizer_file_name(fold_count))

        for key, algorithm in algorithms.items():
            algorithm["train"](processed_data_train, target_train, fold_count)

        test_data = data_frame.iloc[test_index]

        test_data.to_csv(get_test_file_name(fold_count))
        fold_count += 1
