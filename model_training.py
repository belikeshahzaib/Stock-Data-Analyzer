import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def getNumericFeatures(df, excludeTargets=None):
    if excludeTargets is None: excludeTargets = []
    return [c for c in df.columns if c not in excludeTargets and df[c].dtype in [np.float64, np.int64]]

def trainModels(dfTrain, dfTest, nFuture=5, saveModels=True, showFeatureImportance=False):
    dfTrain, dfTest = dfTrain.copy(), dfTest.copy()
    targetCols = [f"Close_plus_{i}" for i in range(1, nFuture + 1)]
    featureCols = getNumericFeatures(dfTrain, excludeTargets=targetCols + ['Close'])
    commonFeatures = sorted(list(set(featureCols).intersection(set(dfTest.columns))))

    xTrain = dfTrain[commonFeatures]
    xTest  = dfTest[commonFeatures]

    scalerX = StandardScaler()
    xTrainScaled = scalerX.fit_transform(xTrain)
    xTestScaled  = scalerX.transform(xTest)

    modelsDict = {}
    predsTest = pd.DataFrame(index=dfTest.index)

    for target in targetCols:
        yTrain = dfTrain[target]
        yTest  = dfTest[target]
        model = XGBRegressor(n_estimators=600, learning_rate=0.035, max_depth=6, random_state=42)
        model.fit(xTrainScaled, yTrain, eval_set=[(xTestScaled, yTest)], verbose=0)
        modelsDict[target] = model
        predsTest[target] = model.predict(xTestScaled)

    comparisonDf = dfTest[['Close'] + targetCols].copy()
    for col in targetCols:
        comparisonDf[col + "_pred"] = predsTest[col].iloc[-1]
    
    return comparisonDf.dropna().reset_index(drop=True)