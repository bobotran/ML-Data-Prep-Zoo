import openml
from sklearn import neighbors, ensemble
from Load_Predictions import *
from downstream_models import *
from Featurize import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
import copy
import warnings
warnings.filterwarnings("ignore")

# print(data.get_data())

import sortinghat.pylib as pl

# Prereqs: pyarrow, Autogluon, nltk, sklearn, 

# rf: Random Forest, neural: Neural Model, knn: K-nn, logreg: Logistic Regression, svm: RBF_SVM

# Numeric': 0,
# Categorical': 1,
# Datetime':2,
# Sentence':3,
# URL': 4,
# Numbers': 5,
# List': 6,
# Unusable': 7,
# Custom Object': 8

idx2label = {
    0 : 'numeric',
    1 : 'categorical',
    2 : 'datetime',
    3 : 'sentence',
    4 : 'url',
    5 : 'embedded-number',
    6: 'list',
    7 : 'not-generalizable',
    8 : 'context-specific'
}

# Convert column to pandas datatype
def get_sortinghat_type(col):
    """ OpenML's conversion from Pandas dtypes to ARFF format
    """
    PD_DTYPES_TO_ARFF_DTYPE = {"integer": "INTEGER", "floating": "REAL", "string": "STRING"}
    attributes_arff = []

    # skipna=True does not infer properly the dtype. The NA values are
    # dropped before the inference instead.
    column_dtype = pd.api.types.infer_dtype(col.dropna(), skipna=False)
    
    if column_dtype == "categorical":
        return 'categorical'
    elif column_dtype == "boolean":
        return 'categorical'
    elif column_dtype in ('integer', 'floating'):
        return 'numeric'
    # elif column_dtype == 'string':
    #     return 'sentence'
    
    return 'context-specific'

# openml.config.apikey = 'YOURKEY'
# CONFIG FILE PATH = C:\Users\Victor\.openml
# openml.config.start_using_configuration_for_example()

# task = openml.tasks.get_task(403) # eeg eye state
# task = openml.tasks.get_task(115) # diabetes

# Keep count of when both declare numeric
numeric_same = 0.0
# Keep count of when both declare categorical
categorical_same = 0.0
# Keep count of when both declare context-specific
context_specific_same = 0.0

total_similarity = 0.0
total_cols = 0
average_similarity = 0.0
successful_tasks = 0
# Iterate over some range of task ids
task_start = 0
task_end = 53
for i in range(task_start, task_end):
    print("Task Id:", i)
    try:
        # Try loading the task
        task = openml.tasks.get_task(i) 
    except KeyboardInterrupt:
        print("Keyboard Interrupt - Exiting")
        sys.exit()
    except:
        print("Dataset not found - moving to next one!")
        continue

    successful_tasks += 1
    # print("task:", task)


    # metadata = task._get_repr_body_fields()
    data = openml.datasets.get_dataset(task.dataset_id)

    X, y, cat_ind, attr_names = data.get_data()


    # Get inferred types for all columns for dataset
    # TODO: fix bug in Load_Predictions.py/FeaturizeFile file trying to convert column
    dataFeaturized = FeaturizeFile(X)

    dataFeaturized1 = ProcessStats(dataFeaturized)
    dataFeaturized2 = FeatureExtraction(dataFeaturized,dataFeaturized1,0)
    dataFeaturized2 = dataFeaturized2.fillna(0)
    rf_infer_types = Load_RF(dataFeaturized2)

    cur_col = 0
    similarity = 0.0
    for (columnName, columnData) in X.iteritems():
        # Get pandas inferred type for the given col
        pd_infer_type = get_sortinghat_type(columnData)
        # Get rf infer type from given label
        rf_infer_type = idx2label[int(rf_infer_types[cur_col])]

        # Check if the pandas inferred type is the same as the random forest inferred type
        if pd_infer_type == rf_infer_type:
            similarity += 1
            if pd_infer_type == "numeric":
                numeric_same += 1
            elif pd_infer_type == "categorical":
                categorical_same += 1
            elif pd_infer_type == "context-specific":
                context_specific_same += 1
        else:
            print(f'-Column:{columnName:20}\tPD_infer:{pd_infer_type:15}\tRF_infer:{rf_infer_type}')
        # print(f'Column Name:{columnName}\tPD_infer:{pd_infer_type}\tRF_infer:{rf_infer_type}')
        cur_col += 1

    total_similarity += similarity
    total_cols += cur_col

    calc_similarity = similarity / cur_col
    print(f"Similarity: {calc_similarity:.4f}")

    average_similarity += calc_similarity

print()
print(f"Successful Tasks:{successful_tasks} out of {task_end - task_start} total tried")
print(f"Total Similarity:{total_similarity / total_cols:.4f}\tSameCols:{total_similarity}\tTotCols:{total_cols}")
print(f"Average Similarity per Dataset:{average_similarity / successful_tasks:.4f}")
print()
print(f"Numeric_same:{numeric_same}\t%:{numeric_same/total_cols:.4f}")
print(f"Categorical_same:{categorical_same}\t%:{categorical_same/total_cols:.4f}")
print(f"ContextSpecific_same:{context_specific_same}\t%:{context_specific_same/total_cols:.4f}")
