import numpy as np   , pandas as pd  
from sklearn.linear_model import LinearRegression as Linreg  , LogisticRegression as Log_reg
from sklearn.metrics import accuracy_score  , silhouette_score , r2_score , f1_score  , precision_score , classification_report
from sklearn.naive_bayes import MultinomialNB  , GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import AdaBoostClassifier  , RandomForestClassifier , RandomForestRegressor , VotingClassifier 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv(r"C:\Users\lap\Downloads\play_tennis_Model\play_tennis_dataset.csv")

# data.info()


def reliable(day:str) :
    day = day.replace('D' , '')
    return day

 
data["Play"] =  data["Play"].map({"Yes" : 1 , "No" : 0 })
data["Outlook"] = data["Outlook"].map( {'Overcast' : 1  ,  'Sunny' : 2  , 'Rainy' : 0})
data["Temperature"] = data["Temperature"].map({'Mild' : 1 , 'Cool' : 0 , 'Hot' :2})
data["Humidity"]  =  data["Humidity"].map({'Normal' :0  ,'High': 1})
data['Wind'] = data['Wind'].map({'Strong' :1 , 'Weak' : 0})


data["Day"] = data["Day"].apply(reliable)

columns_contains_nulls = ["Outlook" , "Temperature" , "Humidity" , "Wind"]

# for column in data.columns :
#         q1 = data[column].quantile(0.25)
#         q3 = data[column].quantile(0.75)
#         iqr = q3 - q1 
#         low = q1 - 1.5 * iqr
#         high = q3 + 1.5 * iqr
#         outlayers = data[data[column] < low | data[column] > high]
#         percentage = float( (outlayers.shape[0] / data.shape[0])*100)
#         if percentage > 7 :
#             # remove the outlayers 
#             data = data[data[column] > low & data[column] < high]
 
for c in columns_contains_nulls :
   data[c] = data[c].fillna(data[c].mean())

data["Play"] = data["Play"].astype("int64")
data["Day"]  = data["Day"].astype("int64")
data["Outlook"] = data["Outlook"].astype("int64") 
data["Temperature"] = data["Temperature"].astype("int64")
data["Humidity"]  =  data["Humidity"].astype("int64")
data['Wind'] = data['Wind'].astype("int64")

x = data.drop(columns=['Play'])
y = data['Play']

x_train , x_test , y_train  , y_test = train_test_split(x , y , test_size=0.2 ,random_state= 42)
# x_train2 , x_test2 , y_train2  , y_test2 = train_test_split(x , y , test_size=0.2 ,random_state= 42)


# here we will use Standard Scalar to make standarization to the data ==> Remember to search about its code 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
model_randomC = RandomForestClassifier()

model_randomC.fit(x_train , y_train)

y_train_pred = model_randomC.predict(x_train)
y_test_pred = model_randomC.predict(x_test)
 
'''print(classification_report(y_train, y_train_pred))
print('--'*50)
print(classification_report(y_test, y_test_pred))
print('##'*50)'''

###  Then the used  model is   RandomForestClassifier


'''with open("model.pkl" , 'wb') as model_file :
    pickle.dump(model_randomC ,model_file )

with open("scaler.pkl" , 'wb') as scaler_file :
    pickle.dump(scaler , scaler_file )
print("Done.....")'''

### Draw correaltion between target and all features 
correlation = data.corr()
plt.figure(figsize=[10 , 8])
sns.heatmap(data= correlation , annot=True , fmt=".2f" , cmap="coolwarm" , robust=True)
plt.title("Showing the correcltion between the target and all features")
# plt.xlabel("The features names")
# plt.ylabel("The features names with Target")

plt.show()

# ----------------------------------------- 

# for drawing scatterplots between all targets 


# long_data = pd.melt(data , id_vars= [data['Play']] , value_vars=x.columns.tolist() , var_name="feature" , value_name=f"value" )

# the drawings net

'''long_data = pd.melt(# melt uses the columns which works with
    data ,
    id_vars= 'Play', # not data['Play']
    value_vars=x.columns.tolist() ,
    var_name='feature' , value_name='value'
)
long_data['type'] = long_data['feature'].map(
    lambda  c : "categorical" if data[c].dtype =="object" else "numerical"
    )
g = sns.FacetGrid(long_data , col='feature' , col_wrap=3 , sharex=False , sharey=False , height=3 )

def facet_plot(x , y , hue , **kwargs):
    if kwargs.get("data")["type"].iolc[0]== "categorical":
        sns.countplot(x=x , hue=hue , **kwargs)
    else :
        sns.boxenplot(x=hue , y=y , **kwargs)

g.map_dataframe(sns.scatterplot , x ='feature' , y='value' , hue='Play'  )

g.set_titles(col_template='Play')
g.add_legend(title="Play")
plt.subplots_adjust(top=0.9)
g.fig.suptitle(f"Relationship between {data['Play']} and other features")

plt.show()
'''


# نحدد عدد الـ features
features = x.columns.tolist()
n = len(features)

# نعمل grid من الـ subplots
rows = (n + 2) // 3   # نخلي كل 3 أعمدة في صف
fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
axes = axes.flatten()

for i, col in enumerate(features):
    ax = axes[i]
   
    if data[col].dtype == 'object':   # categorical
        sns.countplot(data=data, x=col, hue='Play', ax=ax)
        ax.set_title(f"{col} vs Play (countplot)")
    else:   # numerical
        sns.boxplot(data=data, x='Play', y=col, ax=ax)
        ax.set_title(f"{col} vs Play (boxplot)")
   
    ax.set_xlabel(col)
    ax.set_ylabel("")

# لو عدد الرسومات أقل من grid
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Relationship between Play and Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

