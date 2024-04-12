from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import optuna
from optuna import create_study

class ModelTrainer:
    """
    Class to train an XGBoost classifier with hyperparameter optimization using Optuna.

    Args:
    - train_data (pandas.DataFrame): The training dataset.
    - target_col (str): The name of the target column.
    - feature_cols (list or str, optional): List of feature columns or 'auto' to automatically select features. Defaults to 'auto'.
    - cv (int, optional): Number of folds for cross-validation. Defaults to 5.
    - random_state (int, optional): Random state for reproducibility. Defaults to 42.
    """

    def __init__(self, train_data, target_col, feature_cols='auto', cv=5, random_state=42):
        self.train_data = train_data
        self.target_col = target_col
        self.cv = cv
        self.random_state = random_state
        self.best_params = None

        # If feature_cols is set to 'auto', select all columns except 'CandidateID' and target_col
        if feature_cols == 'auto':
            self.feature_cols = [col for col in train_data.columns if col not in ['CandidateID', target_col]]
        else:
            self.feature_cols = feature_cols

        # Initialize StratifiedKFold for cross-validation
        self.kfold = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        
        # Assign fold numbers to each row in train_data
        for i, (trn, val) in enumerate(self.kfold.split(self.train_data, self.train_data[target_col])):
            self.train_data.loc[val, 'kfold'] = i
        self.train_data['kfold'] = self.train_data['kfold'].astype(int)

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
        - trial (optuna.Trial): Optuna trial object.

        Returns:
        - float: Negative average f1 score over all folds.
        """
        f1 = 0
        for fold in range(5):
            xtr, ytr, xval, yval = self._get_train_val_split(fold)

            # Train XGBoost model and calculate f1 score
            _, metrics = self.fit_xgb(trial, xtr, ytr, xval, yval)
            f1 += metrics['valid f1_score'] / 5  # Average f1 score over all folds
        
        return -f1  # Negative because Optuna minimizes the objective function

    def _get_train_val_split(self, fold):
        """
        Get training and validation data for a specific fold.

        Args:
        - fold (int): Fold number.

        Returns:
        - tuple: Tuple containing xtr, ytr, xval, yval.
        """
        trn_idx = self.train_data['kfold'] != fold
        val_idx = self.train_data['kfold'] == fold
        trn = self.train_data.loc[trn_idx, :]
        val = self.train_data.loc[val_idx, :]
        return trn[self.feature_cols].values, trn[self.target_col].values, val[self.feature_cols].values, val[self.target_col].values

    def fit_xgb(self, trial, xtr, ytr, xval, yval):
        """
        Train an XGBoost model and calculate f1 score.

        Args:
        - trial (optuna.Trial): Optuna trial object.
        - xtr, ytr (array-like): Training data and labels.
        - xval, yval (array-like): Validation data and labels.

        Returns:
        - tuple: Tuple containing trained model and dictionary of metrics.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, 100),
            "subsample": trial.suggest_discrete_uniform("subsample", 0.6, 1, 0.1),
            "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.6, 1, 0.1),
            "eta": trial.suggest_loguniform("eta", 1e-3, 0.1),
            "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
            "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        }

        model = XGBClassifier(**params, tree_method='gpu_hist', random_state=self.random_state)
        model.fit(xtr, ytr, eval_metric='mlogloss')
        
        # Predict on training and validation data
        y_tr_pred = model.predict(xtr)
        y_val_pred = model.predict(xval)
        
        # Calculate f1 score
        f1 = {
            "train f1_score": f1_score(ytr, y_tr_pred, average='micro'),
            "valid f1_score": f1_score(yval, y_val_pred, average='micro')
        }
        
        return model, f1

    def optimize_hyperparams(self, n_trials=20):
        """
        Optimize hyperparameters using Optuna.

        Args:
        - n_trials (int, optional): Number of trials for optimization. Defaults to 20.
        """
        study = create_study(direction="minimize", study_name='XGBoost optimization')
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params

    def train_final_model(self):
        """
        Train the final model using the best hyperparameters.

        Returns:
        - XGBClassifier: Trained XGBoost model.
        """
        model = XGBClassifier(**self.best_params, tree_method='gpu_hist', random_state=self.random_state)
        model.fit(self.train_data[self.feature_cols], self.train_data[self.target_col])
        return model
