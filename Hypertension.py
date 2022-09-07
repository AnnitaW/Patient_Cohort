import pandas as pd
import pandas_gbq
import pandas as pd

# Machine Learning
from   sklearn.preprocessing import StandardScaler
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import  accuracy_score, roc_auc_score
from   sklearn.ensemble import RandomForestClassifier
from   imblearn.over_sampling import RandomOverSampler

#data acquisition
from google.cloud import bigquery
%load_ext google.cloud.bigquery
#%reload_ext google.cloud.bigquery

# Set your default project here
pandas_gbq.context.project = 'learnclinicaldatascience'
pandas_gbq.context.dialect = 'standard'

query='''SELECT * 
         FROM course3_data.hypertension_goldstandard
'''
hypert_df=pd.read_gbq(query, project_id='learnclinicaldatascience')

#print(hypert_df.head())
print('This is the size of the hypertension:', hypert_df.shape)

#Pulling the billing code (icd) for hypertension
query =''' SELECT distinct(SUBJECT_ID) 
            FROM mimic3_demo.DIAGNOSES_ICD
            WHERE ICD9_CODE in ("4019", "4011", "4010")
'''
icd_hyp= pd.read_gbq(query, project_id='learnclinicaldatascience')
icd_hyp['icd_hyp']=1

# join both datasets
hypert_df_icd=pd.merge(hypert_df,icd_hyp, on='SUBJECT_ID', how='left').fillna(0).astype(int)

# Looking into medications that are used to treat patients with hyperthension
query = '''SELECT SUBJECT_ID, CONCAT(c.VALUENUM, ' ', c.VALUEUOM) AS BP
          FROM mimic3_demo.CHARTEVENTS c
          WHERE VALUEUOM Like '%mmHg%'

'''
BloodP = pd.read_gbq(query, project_id='learnclinicaldatascience')

BloodP_140 = BloodP[BloodP['BP'] == '140 mmHg']
# BP_140.head()
BloodP_90 = BloodP[BloodP['BP'] == '90 mmHg']
# BP_90.head()

BloodP_OR = BloodP[(BloodP['BP'] == '90 mmHg') | (BloodP['BP'] == '140 mmHg')]

# merge the medication, ICD code, gold standard data
hyper_BP_ICD=pd.merge(hyp_df_icd, BloodP_OR, on='SUBJECT_ID', how='left').fillna(0)

#group the hyper_BP_ICD and count the number of blood pressure measurement for each ID
count_BP=hyper_BP_ICD.groupby(['SUBJECT_ID']).count()
count_BP.rename(columns={'BP': 'Count_BP'}, inplace=True)

#removing the units
hyper_BP_ICD['BP']=hyper_BP_ICD['BP'].map(lambda x: str(x)[:-5])
# converting string into numbers
hyper_BP_ICD['BP']=hyper_BP_ICD['BP'].map(lambda x: int(0) if str(x)=='' else int(x))

mean_BP=hyper_BP_ICD.groupby('SUBJECT_ID')['BP'].mean()

# merging the newly created data
df_merged=pd.merge(mean_BP,count_BP, on='SUBJECT_ID')
df_merged.rename(columns={'BP': 'mean_BP'}, inplace=True)
df_merged.drop(['HYPERTENSION', 'icd_hyp'], axis=1, inplace=True)

# Adding the newly created data to the groundtruth data
data_clean=pd.merge(hypert_df_icd,df_merged, on='SUBJECT_ID')

#Split the X and y dataset
data_drop_target=data_clean.drop(['SUBJECT_ID','HYPERTENSION'], axis=1)
target = data_clean['HYPERTENSION']

#Class Imbalance
# check first if our class is imbalanced
data_clean['HYPERTENSION'].value_counts()

#Perfoming oversample of the minority class
# Oversampling of the minority class hypertension
oversamp = RandomOverSampler()
X_oversamp, y_oversamp = oversamp.fit_resample(data_drop_target, target)

y_oversamp.value_counts()

#Feature Scaling
data_drop_target= X_oversamp
target =y_oversamp

# scaling features
scaler_hyp = StandardScaler()
scaled_data_hyp = scaler_hyp.fit_transform(data_drop_target)
scaled_data_hyp = pd.DataFrame(scaled_data_hyp)
scaled_data_hyp.columns = data_drop_target.columns

#Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_data_hyp, target, test_size = 0.3, random_state = 0)

#Train Model
# Training the model:
model=model = RandomForestClassifier()
model.fit(X_train, y_train)

#Model Predictions
# Predicting our hypertension class
y_pred = model.predict(X_test)

# extracting the probability of the test dataset
y_pred_prob = model.predict_proba(X_test)
y_pred_prob = [x[1] for x in y_pred_prob]

#Model Evaluation
# valuating the model
print("\n Accuracy Score is: \n ",accuracy_score(y_test,y_pred))
print("\n AUC Score is: \n", roc_auc_score(y_test, y_pred_prob))

#Hyperparameter Tuning
# The random forest model gave us a better model
grid_search_random_forest = {'max_depth'   : [10,20,40],
                            'n_estimators' : [100,200,300],
                            'min_samples_leaf' : [1,2,5]
                           }
grid_search_RF= grid_search_random_forest
grid = GridSearchCV(model, grid_search_RF, refit = True, verbose = 3, n_jobs = -1)
# fit the model for grid search
grid.fit(X_train, y_train)

# new predictions
y_pred = grid.predict(X_test)
y_pred_prob = grid.predict_proba(X_test)
y_pred_prob = [x[1] for x in y_pred_prob]

# valuating the model with the best parameter
print("\n Accuracy Score is: \n ",accuracy_score(y_test,y_pred))
print("\n AUC Score is: \n", roc_auc_score(y_test, y_pred_prob))


