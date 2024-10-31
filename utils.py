from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
import sklearn.metrics as skm


# remaps
remap_race={'White':'Sør-Øst', # sør-øst
       'Black':'Midt-Norge', # midt-norge
       'Asian':'Vest', # vest
       'Hispanic':'Nord' # nord
       }

def fetch_data(file_name: str, remap_cols: tuple, remap_vals: dict) -> pd.DataFrame:
       """
       Process the data from a file and perform remapping of columns.

       Args:
              file_name (str): The name of the file to read the data from.
              remap_cols (tuple): A tuple of column names to be remapped.
              remap_vals (dict): A dictionary mapping the original values to the new values.

       Returns:
              pd.DataFrame: The processed data.
       """
       # Read in train and test sets
       data = pd.read_csv(file_name, header=0)
       data[remap_cols[1]] = data[remap_cols[0]].replace(remap_vals)
       data = data.drop(remap_cols[0], axis=1)

       return data



def group_pivot(labelgroup: str, yvalue: str, dataset: pd.DataFrame, SZonly: bool = False) -> pd.DataFrame:
  """
  Formats the data for stacked bar graphs.

  Args:
    labelgroup (str): The column name to group by.
    yvalue (str): The column name for the y-axis values.
    dataset (pd.DataFrame): The dataset to process.
    SZonly (bool, optional): If True, select only columns with a SZ diagnosis. Defaults to False.

  Returns:
    pd.DataFrame: The formatted pivot table.
  """
  if SZonly:
    dataset = dataset.loc[dataset["Diagnosis"] == 0]

  grouped = (
    dataset.groupby([labelgroup])[yvalue]
    .value_counts(normalize=True)
    .rename("percentage")
    .reset_index()
  )

  pivot = pd.pivot_table(
    grouped, index=labelgroup, columns=yvalue, values="percentage", aggfunc="sum"
  )
  return pivot



def preprocess_data(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by separating the features and the target variable.

    Args:
        data (pd.DataFrame): The input data.
        target (str): The name of the target variable.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The preprocessed features and target variable.
    """
    df_x = data.drop(target, axis=1)
    df_y = data[target]
    return df_x, df_y


def onehot(data: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on the specified categorical columns of the input dataframe.

    Args:
        data (pd.DataFrame): The input dataframe.
        categories (List[str]): The list of categorical columns to be one-hot encoded.

    Returns:
        pd.DataFrame: The dataframe with one-hot encoded columns.
    """
    ordinalencoder = OneHotEncoder()
    onehot = ordinalencoder.fit_transform(data[categories])

    columns = []
    for i, values in enumerate(ordinalencoder.categories_):
        for j in values:
            columns.append(str(categories[i] + "_" + j))

    return pd.DataFrame(onehot.toarray(), columns=columns)



def confusion_matrix(truelabels: pd.Series, predictions: np.ndarray) -> None:
    """
    Calculate and display the confusion matrix and performance metrics.
    
    Args:
        truelabels (pd.Series): The true labels.
        predictions (np.ndarray): The predicted labels.

    Returns:
    None
    """
    
    confusion_matrix = skm.confusion_matrix(truelabels, predictions)
    tn, fp, fn, tp = confusion_matrix.ravel()
    print(
        "Sensitivity: ", np.round(tp / (tp + fn), 3),
        "\nSpecificity: ", np.round(tn / (tn + fp),3),
        "\nPPV: ", np.round(tp / (tp + fp),3),
        "\nNPV: ", np.round(tn / (tn + fn),3),
    )

    skm.ConfusionMatrixDisplay(confusion_matrix).plot()


import fairlearn
from fairlearn.metrics import MetricFrame
import fairlearn.metrics
import pandas as pd

def eval_fairness(truelabels: pd.Series, 
                  predictions: pd.Series, 
                  sensitive_var: pd.Series, 
                  sensitive_ref: str, 
                  metric: str) -> pd.DataFrame:
  """
  Calculate fairness metrics for a given metric, false positive rates, and parity.

  Parameters:
    truelabels (pd.Series): True labels of the data.
    predictions (pd.Series): Predicted labels of the data.
    sensitive_variable (pd.Series): Series with the sensitive variables.
    reference (str): Reference group for calculating parity.
    metric (str): Name of the fairness metric.

  Returns:
    pd.DataFrame: Fairness metrics for the given metric, false positive rates, and parity.
  """


  metrics = {"FNR" : fairlearn.metrics.false_negative_rate,
             "FPR" : fairlearn.metrics.false_positive_rate
             }
  
  if metric in metrics.keys():
    selected_metric = metrics[ metric ]
  else:
    print("Metric not valied")
    pass

  fmetrics = MetricFrame(metrics=selected_metric,
               y_true=truelabels,
               y_pred=predictions,
               sensitive_features=sensitive_var)

  results = pd.DataFrame([fmetrics.by_group, fmetrics.by_group / fmetrics.by_group[sensitive_ref]],
               index=[metric, f"{metric} Parity"])
  return results



def intersectionalf(truelabels: pd.Series, 
                    predictions: pd.Series,
                    test: pd.DataFrame,
                    sensitive_var: str, 
                    sensitive_ref: str,
                    intersect_var: str,
                    intersect_ref: str, 
                    metric: str) -> pd.DataFrame:
  """
  Calculate fairness metrics for a given metric, false positive rates, and parity.

  Parameters:
    truelabels (pd.Series): True labels of the data.
    predictions (pd.Series): Predicted labels of the data.
    test (pd.DataFrame): Test dataframe
    sensitive_var (str): Name of the sensitive variable.
    sensitive_ref (str): Reference group for calculating fairness metrics.
    intersect_var (str): Name of the intersecting variable.
    intersect_ref (str): Reference group for calculating fairness metrics within the intersecting variable.
    metric (str): Name of the fairness metric.

  Returns:
    pd.DataFrame: Fairness metrics for the given metric, false positive rates, and parity.
  """
  sensitive = pd.DataFrame(np.stack([test[sensitive_var], test[intersect_var]], axis=1),
               columns=[sensitive_var, intersect_var])

  metrics = {"FNR" : fairlearn.metrics.false_negative_rate,
             "FPR" : fairlearn.metrics.false_positive_rate
             }
  
  if metric in metrics.keys():
    selected_metric = metrics[ metric ]
  else:
    print("Metric not valied")
    pass

  fmetrics = MetricFrame(metrics=selected_metric,
               y_true=truelabels,
               y_pred=predictions,
               sensitive_features=sensitive)

  results = pd.DataFrame([fmetrics.by_group, fmetrics.by_group / fmetrics.by_group[sensitive_ref][intersect_ref]],
               index=[metric, f"{metric} Parity"])
  return results



import matplotlib.pyplot as plt

def plot_base_histogram(data, inspect_var, break_down_on, **kwargs):
  """
  Plot a histogram of a variable in the given data, broken down by another variable.

  Parameters:
    data (pandas.DataFrame): The input data.
    inspect_var (str): The variable to be inspected.
    break_down_on (str): The variable to break down the histogram on.
    **kwargs: Additional keyword arguments.

  Returns:
    None
  """
  # Filter the dataframe by each value of "break_down_on"
  values = data[break_down_on].unique()
  for val in values:
    _data = data[data[break_down_on] == val]
    plt.hist(_data[inspect_var], alpha=0.5, label=val, density=True, bins=kwargs.get('bins', 25))

  plt.xlabel(inspect_var)
  plt.ylabel('Frequency')
  plt.title('Histogram of '+inspect_var+' for each '+break_down_on)
  plt.legend()
  plt.show()