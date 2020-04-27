from at_ml import dataset, lof_lgbm

d=dataset()
m=lof_lgbm()

# X_train, X_test, y_train, y_test = d.get_data()
# m.train(X_train, y_train, X_test, y_test)
m.predict_results(data_predict)