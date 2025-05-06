# -------------------------------------------------------
# -------------------------------------------------------

import pandas as pd
import numpy as np
import os.path as file

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# -------------------------------------------------------
# -------------------------------------------------------


def machineLearningExperiment(current_seed, algorithm):

    output_file = f"output/performances_{algorithm}_seed_{current_seed}.csv"
    if(file.exists(output_file)):
        print(" @ File already exists, skipping...")
        return 

    print(" @ Loading features")

    health_features = "data/features/healthy_rgb.csv"
    osteo_features  = "data/features/diagnosis_rgb.csv"

    df_health = pd.read_csv(health_features)
    df_osteo  = pd.read_csv(osteo_features)

    # -----------------------------
    #  Preprocessing
    # -----------------------------

    # order by filename
    df_health['ids'] = df_health['Name_file'].str.replace("image_", "").str.replace(".png", "")
    df_health['ids'] = pd.to_numeric(df_health['ids'])

    df_osteo['ids'] = df_osteo['Name_file'].str.replace("image_", "").str.replace(".png", "")
    df_osteo['ids'] = pd.to_numeric(df_osteo['ids'])

    # Sorting by ids, the same order in the DL experiments
    df_health = df_health.sort_values(by=['ids'])
    df_osteo  = df_osteo.sort_values(by=['ids'])

    # merge by rows, remove constans columns
    df_all = pd.concat([df_health, df_osteo])

    # Removing constant features
    constant_columns = [col for col in df_all.columns if df_all[col].nunique() == 1]
    constant_columns.append("ids")
    df_cleaned = df_all.drop(columns=constant_columns)

    # Removing highly correlated features ( > 0.95)
    corr_matrix = df_cleaned.drop("Name_file", axis=1).corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop     = [column for column in upper.columns if any(upper[column] > 0.95)]
    df_cleaned.drop(to_drop, axis=1, inplace=True)
    X = df_cleaned

    # ----------------------------
    # Creating labels
    # ----------------------------
    print(" @ Creating labels")

    Y_healthy = np.zeros(df_health.shape[0])
    Y_osteo   = np.ones(df_osteo.shape[0])
    Y = np.concatenate((Y_healthy, Y_osteo))

    # ----------------------------
    # Splitting files into training and testing folds
    # ----------------------------
   
    print(" @ Splitting data into training and testing folds")
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=current_seed, stratify=Y)

    # ----------------------------
    # ----------------------------
    
    print(f" @ Training: {algorithm}")
  
    model = None
    match algorithm:
        case "rf":      model = RandomForestClassifier(n_estimators=100, random_state=current_seed)
        case "dt":      model = DecisionTreeClassifier(random_state=current_seed)
        case "bagging": model = BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                              n_estimators=100, random_state=current_seed)
        case "knn":     model = make_pipeline(StandardScaler(), KNeighborsClassifier())
        case "svm":     model = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=current_seed))
        case "ridge":   model = make_pipeline(StandardScaler(), RidgeClassifier(random_state=current_seed))
        case _:        raise ValueError(f"Unknown algorithm: {algorithm}")
                    
    # ----------------------------
    # ----------------------------

    true_model = model.fit(x_train.drop("Name_file", axis = 1), y_train)
    predictions = true_model.predict(x_test.drop("Name_file", axis = 1))

    # ----------------------------
    # ----------------------------

    print("----------------------------")
    acc = accuracy_score(y_test, predictions)
    bac = balanced_accuracy_score(y_test, predictions)
    f1s = f1_score(y_test, predictions)
    print("acc = ", acc)
    print("bac = ", bac)
    print("f1c = ", f1s)
    print("----------------------------")

    print(" @ Exporting results") 

    preds = pd.DataFrame(predictions, index = x_test.index)
    preds = preds.rename(columns={0: 'predictions'})

    # creating a data frame with [img_path, seed, Y, prediction, algo]
    preds['algo'] = algorithm
    preds['seed'] = current_seed
    preds['Y'] = y_test
    preds['x_test_file'] = x_test[["Name_file"]]
    preds.to_csv(f"output/predictions_{algorithm}_seed_{current_seed}.csv", index = False)

    performances = []
    performances.append([acc, bac, f1s, current_seed])
    df_performances = pd.DataFrame(performances)
    df_performances.columns = ['accuracy', 'balanced_accuracy', 'fscore', 'seed']

    df_performances.to_csv(output_file, index = False)
 
    print(" Finished !!! :) ")
    print(" ----------------------------")


# -------------------------------------------------------
# -------------------------------------------------------
