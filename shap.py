import numpy as np
import random
import itertools


def get_custom_sample(data_without_label, row_index, features_to_keep):
    new_sample = np.empty(data_without_label.shape[1])

    for feature in features_to_keep:
        new_sample[feature] = data_without_label.iloc[row_index, feature]

    for i in range(len(new_sample)):
        if i not in features_to_keep:
            random_row = random.choice(range(len(data_without_label)))
            new_sample[i] = data_without_label.iloc[random_row, i]

    return new_sample


def get_null_prediction(
    model, data_without_label, sample_size_for_null_prediction, seed=None
):
    sample_size_for_null_prediction = min(
        sample_size_for_null_prediction, len(data_without_label)
    )
    sample_df = data_without_label.sample(
        sample_size_for_null_prediction, random_state=seed
    )
    predictions = model.predict(sample_df)
    return predictions.mean()


class SHAP:
    def __init__(
        self,
        model,
        data_without_label,
        row_index,
        null_prediction=None,
        sample_size_for_null_prediction=10000,
        sample_size_for_expected_prediction=1000,
    ):
        if null_prediction:
            self.null_prediction = null_prediction
        else:
            self.null_prediction = get_null_prediction(
                model, data_without_label, sample_size_for_null_prediction
            )
        self.true_prediction = model.predict(
            data_without_label.loc[row_index].values.reshape(1, -1)
        )[0]
        self.column_indices = list(range(len(data_without_label.columns)))
        self.set_expected_predictions(
            model, data_without_label, row_index, sample_size_for_expected_prediction
        )

    def set_expected_predictions(
        self, model, data_without_label, row_index, sample_size_for_expected_prediction
    ):
        self.expected_predictions = {}
        self.expected_predictions[frozenset(self.column_indices)] = self.true_prediction
        for permutation in itertools.permutations(self.column_indices):
            for i in range(1, len(permutation)):
                custom_samples = []
                features_to_keep = frozenset(permutation[:i])
                if features_to_keep not in self.expected_predictions:
                    for _ in range(sample_size_for_expected_prediction):
                        custom_samples.append(
                            get_custom_sample(
                                data_without_label, row_index, features_to_keep
                            )
                        )
                    expected_prediction = np.mean(model.predict(custom_samples))
                    self.expected_predictions[features_to_keep] = expected_prediction

    def get_shap(self):
        marginal_contributions = {
            column_index: [] for column_index in self.column_indices
        }
        for permutation in itertools.permutations(self.column_indices):
            for i in range(1, len(permutation) + 1):
                sub_permutation = permutation[:i]
                if len(sub_permutation) == 1:
                    marginal_contributions[sub_permutation[-1]].append(
                        self.expected_predictions[frozenset(sub_permutation)]
                        - self.null_prediction
                    )
                else:
                    marginal_contributions[sub_permutation[-1]].append(
                        self.expected_predictions[frozenset(sub_permutation)]
                        - self.expected_predictions[frozenset((sub_permutation[:-1]))]
                    )
        return [np.mean(marginal_contributions[i]) for i in marginal_contributions]

    def get_shap_interactions(self):
        marginal_contributions = {}
        for permutation in itertools.permutations(self.column_indices):
            for pair in itertools.combinations(permutation, 2):
                if pair not in marginal_contributions:
                    marginal_contributions[pair] = []
                for r in range(len(permutation) - 1):
                    for sub_permutation in itertools.combinations(permutation, r):
                        if (
                            pair[0] not in sub_permutation
                            and pair[1] not in sub_permutation
                        ):
                            if len(sub_permutation) == 0:
                                marginal_contributions[pair].append(
                                    self.expected_predictions[
                                        frozenset(sub_permutation + pair)
                                    ]
                                    - self.expected_predictions[
                                        frozenset(sub_permutation + (pair[0],))
                                    ]
                                    - self.expected_predictions[
                                        frozenset(sub_permutation + (pair[1],))
                                    ]
                                    + self.null_prediction
                                )
                            else:
                                marginal_contributions[pair].append(
                                    self.expected_predictions[
                                        frozenset(sub_permutation + pair)
                                    ]
                                    - self.expected_predictions[
                                        frozenset(sub_permutation + (pair[0],))
                                    ]
                                    - self.expected_predictions[
                                        frozenset(sub_permutation + (pair[1],))
                                    ]
                                    + self.expected_predictions[
                                        frozenset(sub_permutation)
                                    ]
                                )
        return {
            pair: np.mean(marginal_contributions[pair])
            for pair in marginal_contributions
        }
