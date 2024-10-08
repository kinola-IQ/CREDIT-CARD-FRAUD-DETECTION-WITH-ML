
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from imblearn.combine import SMOTETomek
import numpy as np

def tuning(X,y,n_trial=100):
    def objective(trial):
        param ={
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 15),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.2)

            }
        
        #defning model
        model = GradientBoostingClassifier(**param,random_state=0)
        
        
        # using Stratefied Kfold for evaluation
        skf = StratifiedKFold(n_splits = 6 ,random_state=0,shuffle=True)
        score = []
        for train_index,test_index in skf.split(X,y):
            x1_train, x1_test = X.iloc[train_index], X.iloc[test_index]
            y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]
            
            x1_train_res, y1_train_res = SMOTETomek(sampling_strategy=0.4, random_state=0).fit_resample(x1_train, y1_train)

        
            model.fit(x1_train,y1_train)
            predict = model.predict(x1_test)
            performance = matthews_corrcoef(y1_test, predict)
            score.append(performance)
            
        final_score = np.mean(score)
        return final_score
    
    #adding pruner to the study as an equivalent to ealy stopping round
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction='maximize',sampler= optuna.samplers.RandomSampler(seed=0),pruner =pruner)
    study.optimize(objective,n_trial=n_trial,n_jobs=-1)
    
    best_param = study.best_params
    best_score = study.best_value
    best_trial = study.best_trial
    
    return best_param,best_score,best_trial

def skf_val_score(model,X,y,scoring=None,splits=6):
    skf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=0)
    score =[]
    
    for train_index,test_index in skf.split(X, y):
        x1_train,x1_test = X.iloc[train_index],X.iloc[test_index]
        y1_train,y1_test = y.iloc[train_index],y.iloc[test_index]
        
        model.fit(x1_train,y1_train)
        y_pred = model.predict(x1_test)
        if scoring==None:
            performance = matthews_corrcoef(y1_test,y_pred)
        else:
            performance = scoring(y1_test,y_pred)
        
        score.append(performance)
    avg = np.mean(score)
    return avg