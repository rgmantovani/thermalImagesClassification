# -------------------------------------------------------
# -------------------------------------------------------

import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# -------------------------------------------------------
# -------------------------------------------------------

if __name__ == "__main__":

    SEEDS = [404, 666, 42, 171, 51]
    current_seed = SEEDS[4]

    np.random.seed(current_seed)
    random.seed(current_seed)

    print("Seed: ", current_seed)

    # ----------------------------
    # ----------------------------
    
    print(" @ Loading features")

    health_features = "data/features/healthy_rgb.csv"
    osteo_features  = "data/features/diagnosis_rgb.csv"

    df_health = pd.read_csv(health_features)
    df_osteo  = pd.read_csv(osteo_features)

    # order by filename
    # TODO: não estão na ordem correta (está afaltabetica, e nao pelo id)
    df_health = df_health.sort_values(by=['Name_file'])
    df_osteo  = df_osteo.sort_values(by=['Name_file'])

    # merge by rows, remove constans columns
    df_all = pd.concat([df_health, df_osteo])

    constant_columns = [col for col in df_all.columns if df_all[col].nunique() == 1]
    df_cleaned = df_all.drop(columns=constant_columns)
    X = df_cleaned

    # ----------------------------
    # Creating labels
    # ----------------------------
    print(" @ Creating labels")

    #TODO: check Y size
    Y_healthy = np.zeros(df_health.shape[0])
    Y_osteo   = np.ones(df_osteo.shape[0])
    Y = np.concatenate((Y_healthy, Y_osteo))

    # ----------------------------
    # Splitting files into training and testing folds
    # ----------------------------
   
    print(" @ Splitting data into training and testing folds")
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=current_seed, stratify=Y)

    print(" @ Training: Random Forest")
    x_train.columns = x_train.columns.str.strip()
    x_test.columns = x_test.columns.str.strip()

    model = RandomForestClassifier(n_estimators=100, random_state=current_seed)
    true_model = model.fit(x_train.drop("Name_file", axis = 1), y_train)
    predictions = true_model.predict(x_test.drop("Name_file", axis = 1))

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
    preds['algo'] = "RandomForest"
    preds['seed'] = current_seed
    preds['Y'] = y_test
    preds['x_test_file'] = x_test[["Name_file"]]
    preds.to_csv(f"output/predictions_RF_seed_{current_seed}.csv", index = False)

    performances = []
    performances.append([acc, bac, f1s, current_seed])
    df_performances = pd.DataFrame(performances)
    df_performances.columns = ['accuracy', 'balanced_accuracy', 'fscore', 'seed']
    df_performances.to_csv(f"output/performances_RF_seed_{current_seed}.csv", index = False)
 
    print(" Finished !!! :) ")
    print(" ----------------------------")
    
# -------------------------------------------------------
# -------------------------------------------------------
