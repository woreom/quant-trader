import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error

from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
import optuna

import joblib 
import matplotlib.pyplot as plt


def mlp_regressor(inputs, targets, inputs_forecast, save_name, hyper_tune=False, plotResults=True, n_trials=100):

    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=1 - train_ratio, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42, shuffle=False)
    
    def CostFunc(trial):
        param = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
                                                            [
                                                             (256, ), (256, 128), (256, 128, 64),
                                                             (128,), (128,64), (128,64,32),
                                                             (100,), (100,100), (100,50), (64,), (32,),
                                                             (20,10), (20,), (10,)]),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1),
            'max_iter': 100,
            'early_stopping':True,
            'random_state': 42
        }
    
        ## Train Model with Optimal Hyper Parameters
        model = MLPRegressor(**param)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
    
        return rmse
    
    if hyper_tune:
        study = optuna.create_study(direction='minimize', study_name='mlp Tuner')
        study.optimize(CostFunc, n_trials=n_trials)
        params = study.best_params
        params['max_iter'] =1000
        params['early_stopping']=True
        params['random_state']= 42
        
        
        if plotResults:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            optuna.visualization.matplotlib.plot_param_importances(study)
    
        joblib.dump(params, f'params/mlp_regressor_hyper_params_{save_name}.pkl')
    else:
        try:
            # Load the optimal hyperparameters from a previous run
            params = joblib.load(f'params/mlp_regressor_hyper_params_{save_name}.pkl')
        except FileNotFoundError:
            print("Optimal hyperparameters not found. Using default values.")
            params = {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 200,
                'early_stopping':True,
                'random_state': 42
            }
    
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    
    TrainOutputs = model.predict(X_train)
    ValOutputs = model.predict(X_val)
    TestOutputs = model.predict(X_test)
    
    TrainRMSE = mean_squared_error(y_train, TrainOutputs, squared=False)
    ValRMSE = mean_squared_error(y_val, ValOutputs, squared=False)
    TestRMSE = mean_squared_error(y_test, TestOutputs, squared=False)
    
    print('------Model Performance------')
    print('Train RMSE:', TrainRMSE, '| Validation RMSE:', ValRMSE, '| Test RMSE:', TestRMSE)
    
    if plotResults:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        axs[0].plot(y_train, label='Targets')
        axs[0].plot(TrainOutputs, label='Outputs')
        axs[0].set_ylabel('Train')
        axs[0].legend()
    
        axs[1].plot(y_val, label='Targets')
        axs[1].plot(ValOutputs, label='Outputs')
        axs[1].set_ylabel('Validation')
        axs[1].legend()
    
        axs[2].plot(y_test, label='Targets')
        axs[2].plot(TestOutputs, label='Outputs')
        axs[2].set_ylabel('Test')
        axs[2].legend()
    
        plt.show()
    
    rmse = {'Train': TrainRMSE, 'Validation': ValRMSE, 'Test': TestRMSE}
    
    forecast = model.predict(np.reshape(inputs_forecast, (1, -1)))[0]
    
    return forecast, model, rmse



def lgb_regressor(inputs, targets, inputs_forecast, save_name, hyper_tune=False, plotResults=True, n_trials=100):



    # Split the data into training, validation, and test sets
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio =0.1

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=1 - train_ratio, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42, shuffle=False)

    def CostFunc(trial):

        param = {  
            'metric': 'rmse',
            'verbose':-1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'goss']),
            'min_child_samples':trial.suggest_categorical('min_child_samples', [10, 20, 50, 80, 100, 150, 300, 500]),
            'num_leaves': trial.suggest_categorical('num_leaves', [80, 100, 150, 200, 250, 300, 400, 500, 1000, 1250, 1500, 2000, 2500]),
            'max_depth': trial.suggest_categorical('max_depth', [-1,  3, 5, 7, 10, 12, 15, 20]),
            'learning_rate': trial.suggest_categorical('learning_rate', [ 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]),
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 80, 100, 200, 300, 500, 600, 700, 800, 900, 1000, 1200]),
            'reg_alpha': trial.suggest_int('reg_alpha', 0, 100),
            'reg_lambda': trial.suggest_int('reg_lambda', 0, 100),
            'min_split_gain': trial.suggest_int('min_split_gain', 0, 20),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]),
        }

        ## Train Model with Optimal Hyper Parameters
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(param,
              train_data,
              num_boost_round=1000,
              valid_sets=[val_data],
              callbacks=[lgb.early_stopping(stopping_rounds=15)])
        
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        return rmse
        

    if hyper_tune==True:
        study = optuna.create_study(direction='minimize', study_name='lgb Tuner' )
        study.optimize(CostFunc, n_trials=n_trials)
        params=study.best_params 
        if plotResults:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            optuna.visualization.matplotlib.plot_param_importances(study)

        joblib.dump(params, f'params/lgb_regressor_hyper_params_{save_name}.pkl')
    else:
        try:
            # Load the optimal hyperparameters from a previous run
            params = joblib.load(f'params/lgb_regressor_hyper_params_{save_name}.pkl')
        except FileNotFoundError:
            print("Optimal hyper parameters not found. Using default values.")
            params = {
                'boosting_type': 'dart',
            }
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val)
    params['verbose']=-1
    params['metric']='rmse'
    
    model = lgb.train(params,
          train_data,
          num_boost_round=1000,
          valid_sets=[val_data],
          callbacks=[lgb.early_stopping(stopping_rounds=15)])

    TrainOutputs = model.predict(X_train, num_iteration=model.best_iteration)
    ValOutputs=model.predict(X_val, num_iteration=model.best_iteration)
    TestOutputs = model.predict(X_test, num_iteration=model.best_iteration)
    
    TrainRMSE= mean_squared_error(y_train, TrainOutputs, squared=False)
    ValRMSE= mean_squared_error(y_val, ValOutputs, squared=False)
    TestRMSE= mean_squared_error(y_test, TestOutputs, squared=False)
    
    print('------Model Performance------')
    print('Train RMSE:', TrainRMSE, '| Validation RMSE:', ValRMSE, '| Test RMSE:', TestRMSE)
    
    if plotResults:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        axs[0].plot(y_train, label='Targets')
        axs[0].plot(TrainOutputs, label='Outputs')
        axs[0].set_ylabel('Train')
        axs[0].legend()
        
        axs[1].plot(y_val, label='Targets')
        axs[1].plot(ValOutputs, label='Outputs')
        axs[1].set_ylabel('Validation')
        axs[1].legend()
        
        
        axs[2].plot(y_test, label='Targets')
        axs[2].plot(TestOutputs, label='Outputs')
        axs[2].set_ylabel('Test')
        axs[2].legend()
        
        plt.show()

    
    rmse ={'Train': TrainRMSE , 'Validation': ValRMSE , 'Test': TestRMSE}
    
    forecast = model.predict(np.reshape(inputs_forecast, (1,-1)))[0]
    
    return forecast, model, rmse

