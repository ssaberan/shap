import numpy as np
import random
import itertools


def get_custom_sample(data_without_label, row_index, features_to_keep):
    """
    Generates a new sample by keeping the specified features from a given row and randomly
    selecting other features from different rows in the dataset.

    Parameters:
    - data_without_label (DataFrame): The dataset without the label column.
    - row_index (int): The index of the row from which to keep the specified features.
    - features_to_keep (list of int): Indices of the features to keep from the row.

    Returns:
    - new_sample (numpy array): The newly generated sample.
    """
    new_sample = np.empty(data_without_label.shape[1])

    # Copy specified features from the given row.
    for feature in features_to_keep:
        new_sample[feature] = data_without_label.iloc[row_index, feature]

    # Randomly select other features from different rows.
    for i in range(len(new_sample)):
        if i not in features_to_keep:
            random_row = random.choice(range(len(data_without_label)))
            new_sample[i] = data_without_label.iloc[random_row, i]

    return new_sample


def get_null_prediction(
    model, data_without_label, sample_size_for_null_prediction, seed=None
):
    """
    Calculates the average prediction of the model when no specific feature information is provided.
    This serves as a baseline or reference prediction.

    Parameters:
    - model: The prediction model.
    - data_without_label (DataFrame): The dataset without the label column.
    - sample_size_for_null_prediction (int): The number of samples to use for computing the null prediction.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - Average prediction (float) over the randomly sampled data.
    """
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
        """
        Initializes the SHAP class to compute SHAP values for a specific row in the dataset.

        Parameters:
        - model: The prediction model.
        - data_without_label (DataFrame): The dataset without the label column.
        - row_index (int): The index of the row for which to compute SHAP values.
        - null_prediction (float, optional): Precomputed null prediction. If not provided, it's calculated.
        - sample_size_for_null_prediction (int): Number of samples for null prediction calculation.
        - sample_size_for_expected_prediction (int): Number of samples for expected prediction calculation.
        """
        # Set null prediction
        if null_prediction:
            self.null_prediction = null_prediction
        else:
            self.null_prediction = get_null_prediction(
                model, data_without_label, sample_size_for_null_prediction
            )

        # Compute true prediction for the specific row
        self.true_prediction = model.predict(
            data_without_label.loc[row_index].values.reshape(1, -1)
        )[0]

        # Store column indices
        self.column_indices = list(range(len(data_without_label.columns)))

        # Calculate expected predictions for all feature combinations
        self.set_expected_predictions(
            model, data_without_label, row_index, sample_size_for_expected_prediction
        )

    def set_expected_predictions(
        self, model, data_without_label, row_index, sample_size_for_expected_prediction
    ):
        """
        Computes expected predictions for various subsets of features.

        Parameters:
        - model: The prediction model.
        - data_without_label (DataFrame): The dataset without the label column.
        - row_index (int): The index of the row for which to compute expected predictions.
        - sample_size_for_expected_prediction (int): Number of samples to use for each subset of features.
        """
        self.expected_predictions = {}
        self.expected_predictions[frozenset(self.column_indices)] = self.true_prediction

        # Iterate over all permutations of features
        for permutation in itertools.permutations(self.column_indices):
            for i in range(1, len(permutation)):
                custom_samples = []
                features_to_keep = frozenset(permutation[:i])

                # Compute expected prediction if not already done
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
        """
        Computes the SHAP values for each feature.

        Returns:
        - List of SHAP values corresponding to each feature.
        """
        marginal_contributions = {
            column_index: [] for column_index in self.column_indices
        }

        # Consider marginal contributions of features in every permutation
        for permutation in itertools.permutations(self.column_indices):
            for i in range(1, len(permutation) + 1):
                sub_permutation = permutation[:i]

                if len(sub_permutation) == 1:
                    # Contribution of a feature in the context of no other features
                    marginal_contributions[sub_permutation[-1]].append(
                        self.expected_predictions[frozenset(sub_permutation)]
                        - self.null_prediction
                    )
                else:
                    # Contribution of a feature in the context of other features
                    marginal_contributions[sub_permutation[-1]].append(
                        self.expected_predictions[frozenset(sub_permutation)]
                        - self.expected_predictions[frozenset((sub_permutation[:-1]))]
                    )

        # Average the contributions for each feature
        return [np.mean(marginal_contributions[i]) for i in marginal_contributions]

    def get_shap_interactions(self):
        """
        Computes SHAP interaction values for each pair of features.

        Returns:
        - Dictionary of SHAP interaction values for each pair of features.
        """
        marginal_contributions = {}

        # Consider marginal contributions of feature pairs in every permutation
        for permutation in itertools.permutations(self.column_indices):
            for pair in itertools.combinations(permutation, 2):
                if pair not in marginal_contributions:
                    marginal_contributions[pair] = []

                # Calculate contribution of each pair in different contexts, where we consider
                # every subset of the permutation which does not contain the pair as our context.
                for r in range(len(permutation) - 1):
                    for sub_permutation in itertools.combinations(permutation, r):
                        if (
                            pair[0] not in sub_permutation
                            and pair[1] not in sub_permutation
                        ):
                            if len(sub_permutation) == 0:
                                # Contribution of a pair of features in the context of no other features
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
                                # Contribution of a pair of features in the context of other features
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

        # Average the contributions for each pair
        return {
            pair: np.mean(marginal_contributions[pair])
            for pair in marginal_contributions
        }
