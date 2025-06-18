# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math as math
import random
from sklearn.utils import shuffle
import sklearn.preprocessing  as prepro
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import resample
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
#from adjustText import adjust_text

app = Flask(__name__)

# Constants
random_state = 42
plotColor = ['b','g','r','m','c', 'y']
markers = ['+','o','*','^','v','>','<']

# Set up


# Create Data class which has load_data method
class Data:
    def __init__(self):
        print("Data object initiated")
    
    def Load_data(self, filepath, format='csv'):
        """ Read data from file and return data """
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'xlsx':
            return pd.read_excel(filepath)

@app.route('/')
def index():
    return render_template('index.html')
#############################################################################################

@app.route('/load_data')
def load_data():
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    df_head = df_raw.head(10).to_html()

    # Create the output string with HTML formatting
    output = f"<h1>Data Loaded</h1>{df_head}"
    
    # Render the template with the output
    return render_template('index.html', output=output)
#############################################################################################

@app.route('/feature_exploration')
def feature_exploration():
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    
    # Perform feature exploration
    df_shape = f"<strong>Shape of DataFrame:</strong><br>{df_raw.shape[0]} rows, {df_raw.shape[1]} columns"
    df_columns = f"<strong>Columns:</strong><br>{', '.join(df_raw.columns)}"
    df_null = f"<strong>Missing Values:</strong><br>{df_raw.isnull().any().to_string().replace('<', '&lt;').replace('>', '&gt;')}"
    df_duplicated = f"<strong>Duplicated Rows:</strong><br>{df_raw.duplicated().any()}"
    df_dtypes = f"<strong>Data Types:</strong><br>{df_raw.dtypes.to_string().replace('<', '&lt;').replace('>', '&gt;')}"
 
    output = f"<h1>FEATURE EXPLORATION</h1> <p>{df_shape}</p><p>{df_columns}</p><p>{df_null}</p><p>{df_duplicated}</p><p>{df_dtypes}</p>"
    return render_template('index.html', output=output)
#############################################################################################

def Data_transformation_renaming(df_raw):
    df_cal = df_raw.copy()
    
    df_cal.rename(columns={'gender':'Gender'
                       ,'customerID':'CustomerID'
                       ,'Contract':'ContractType'
                       ,'InternetService':'InternetServiceType'
                       ,'tenure':'Tenure'
                      }
              ,inplace=True)


    df_cal['Partner'] = df_cal.Partner.map({'Yes':1,'No':0})
    df_cal['Dependents'] = df_cal.Dependents.map({'Yes':1,'No':0})

    df_cal['PhoneService'] = df_cal.PhoneService.map({'Yes':1,'No':0})
    df_cal['MultipleLines'] = df_cal.MultipleLines.map({'Yes':1,'No':0,'No phone service':0})

    df_cal['InternetService'] = df_cal.InternetServiceType.map({'DSL':1,'Fiber optic':1,'No':0})
    df_cal['OnlineSecurity'] = df_cal.OnlineSecurity.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['OnlineBackup'] = df_cal.OnlineBackup.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['DeviceProtection'] = df_cal.DeviceProtection.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['TechSupport'] = df_cal.TechSupport.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['StreamingTV'] = df_cal.StreamingTV.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['StreamingMovies'] = df_cal.StreamingMovies.map({'Yes':1,'No':0,'No internet service':0})
    df_cal['PaperlessBilling'] = df_cal.PaperlessBilling.map({'Yes':1,'No':0})
    df_cal['Churn'] = df_cal.Churn.map({'Yes':1,'No':0})
    
    # Data mining
    df_cal['IsContracted'] = df_cal.ContractType.map({'One year':1,'Two year':1,'Month-to-month':0})
    
    # Data transformation
    
    # Converting TotalCharges into Numeric, but some of the records are empty, so first we need to deal with them.
    df_cal.loc[df_cal['TotalCharges']==' ','TotalCharges'] = np.nan
    
    # First we convert TotalCharges to float and then replace with tenure * monthly charges
    df_cal['TotalCharges'] = df_cal['TotalCharges'].astype('float64')
    df_cal.loc[df_cal['TotalCharges'].isnull()==True,'TotalCharges'] = df_cal['MonthlyCharges'] * df_cal['Tenure']
    

    return df_cal
data_file = 'files_for_training_model/telcom.csv'
extension = 'csv'
data = Data()
df_raw = data.Load_data(data_file, extension)

df_cal = Data_transformation_renaming(df_raw)
cat_cols = ["Gender","Partner","Dependents","SeniorCitizen","PhoneService","MultipleLines"
                    ,"InternetServiceType","OnlineSecurity","OnlineBackup","DeviceProtection"
                    ,"TechSupport","StreamingTV","StreamingMovies","IsContracted"
                    ,"ContractType","PaperlessBilling","PaymentMethod"]


num_cols = ["Tenure","MonthlyCharges","TotalCharges"]

target_col = 'Churn'

# spliting categorical columns into Nominal and Binary columns

nominal_cols = ['Gender','InternetServiceType','PaymentMethod','ContractType']

binary_cols = ['SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity'
               ,'OnlineBackup' ,'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies'
               ,'PaperlessBilling','InternetService', 'IsContracted']
df_cal[cat_cols].describe(include='all')
df_cal[num_cols].describe()
def Outlier_boxplot(df, col):
    """ Display boxplot for given column """
    sns.boxplot(x=df[col])
    plt.show()


from flask import send_file
#############################################################################################
@app.route('/outlier_detection')
def outlier_detection():
    image_paths = []
    for i in range(3):
        # Create boxplot and save it as a temporary file
        Outlier_boxplot(df_cal, num_cols[i])
        image_path = f'temp_Figure_{i+1}.png'
        plt.savefig(image_path)  # Save the plot to a file
        plt.close()  # Close the plot to avoid overlapping
        image_paths.append(image_path)

    # Render the template with the list of image paths
    return render_template('index.html', image_paths=image_paths)

#############################################################################################
# Import necessary libraries
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Your existing Flask route for churn rate analysis
@app.route('/churn_rate')
def churn_rate():
    """ Rename column names and transformed into proper format and return dataframe """
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    df_cal = df_raw.copy()

    df_cal.rename(columns={'gender': 'Gender',
                           'customerID': 'CustomerID',
                           'Contract': 'ContractType',
                           'InternetService': 'InternetServiceType',
                           'tenure': 'Tenure'},
                  inplace=True)

    df_cal['Partner'] = df_cal.Partner.map({'Yes': 1, 'No': 0})
    df_cal['Dependents'] = df_cal.Dependents.map({'Yes': 1, 'No': 0})

    df_cal['PhoneService'] = df_cal.PhoneService.map({'Yes': 1, 'No': 0})
    df_cal['MultipleLines'] = df_cal.MultipleLines.map({'Yes': 1, 'No': 0, 'No phone service': 0})

    df_cal['InternetService'] = df_cal.InternetServiceType.map({'DSL': 1, 'Fiber optic': 1, 'No': 0})
    df_cal['OnlineSecurity'] = df_cal.OnlineSecurity.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['OnlineBackup'] = df_cal.OnlineBackup.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['DeviceProtection'] = df_cal.DeviceProtection.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['TechSupport'] = df_cal.TechSupport.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['StreamingTV'] = df_cal.StreamingTV.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['StreamingMovies'] = df_cal.StreamingMovies.map({'Yes': 1, 'No': 0, 'No internet service': 0})
    df_cal['PaperlessBilling'] = df_cal.PaperlessBilling.map({'Yes': 1, 'No': 0})
    df_cal['Churn'] = df_cal.Churn.map({'Yes': 1, 'No': 0})

    # Data mining
    df_cal['IsContracted'] = df_cal.ContractType.map({'One year': 1, 'Two year': 1, 'Month-to-month': 0})

    # Data transformation

    # Converting TotalCharges into Numeric, but some of the records are empty, so first we need to deal with them.
    df_cal.loc[df_cal['TotalCharges'] == ' ', 'TotalCharges'] = np.nan

    # First we convert TotalCharges to float and then replace with tenure * monthly charges
    df_cal['TotalCharges'] = df_cal['TotalCharges'].astype('float64')
    df_cal.loc[df_cal['TotalCharges'].isnull() == True, 'TotalCharges'] = df_cal['MonthlyCharges'] * df_cal['Tenure']

    df_cal = Data_transformation_renaming(df_raw)
    cat_cols = ["Gender", "Partner", "Dependents", "SeniorCitizen", "PhoneService", "MultipleLines",
                "InternetServiceType", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "IsContracted",
                "ContractType", "PaperlessBilling", "PaymentMethod"]

    num_cols = ["Tenure", "MonthlyCharges", "TotalCharges"]

    target_col = 'Churn'

    # spliting categorical columns into Nominal and Binary columns

    nominal_cols = ['Gender', 'InternetServiceType', 'PaymentMethod', 'ContractType']

    binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'PaperlessBilling', 'InternetService', 'IsContracted']

    # Read the image as binary data
    with open("churn_rate.png", 'rb') as f:
        image_data = f.read()

    # Encode the binary image data as base64
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    # Generate HTML code to display the image
    image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Churn Rate Analysis">'

    # Set the output variable to the HTML code for the image
    output = f"<h1>Churn Rate Analysis</h1>{image_html}"

    # Render the template with the churn rate analysis image
    return render_template('index.html', output=output)



#############################################################################################
from flask import render_template, url_for

@app.route('/correlation_heatmap')
def correlation_heatmap():
    import base64

    with open("corr.png", 'rb') as f:
        image_data = f.read()

    encoded_image = base64.b64encode(image_data).decode('utf-8')

    image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Correlation Heatmap">'

    output = f"<h1>Correlation Heatmap</h1>{image_html}"

    return render_template('index.html', output=output)


#############################################################################################
def Create_data_label(ax):
    """ Display data label for given axis """
    for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/ 2
                    , bar.get_height() + 0.01
                    , str(round(100 * bar.get_height(),2)) + '%'
                    , ha = 'center'
                    , fontsize = 13)
            
            
def Categorical_var_churn_dist(data, cols, distribution_col):
    """ Distribution of categorical variable based on target variable """
    
    for i,feature in enumerate(cols):
        
        feature_summary = data[feature].value_counts(normalize=True).reset_index(name='Percentage')
        
        plt_cat = sns.catplot(x=feature
                , y='Percentage'
                , data = feature_summary
                , col=distribution_col
                , kind='bar'
                , aspect = 0.8
                , palette = plotColor
                , alpha = 0.6)
        
        if feature == 'PaymentMethod':
            plt_cat.set_xticklabels(rotation= 65, horizontalalignment = 'right')
        
        
        for ax1, ax2 in plt_cat.axes:
            Create_data_label(ax1)
            Create_data_label(ax2)
        
        
        plt.ylim(top=1)
        plt.subplots_adjust(top = 0.9)
        plt.gcf().suptitle(feature+" distribution",fontsize=14)
    plt.show()




@app.route('/categorical_churn_distribution')
def categorical_churn_distribution():
    # Load data and preprocess
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    df_cal = Data_transformation_renaming(df_raw)  # Assuming preprocess_data function is defined

    # Group by churn
    churn_summary = df_cal.groupby('Churn')

    # Plot categorical variable churn distribution
    Categorical_var_churn_dist(churn_summary, cat_cols, 'Churn')

    # Return a message indicating the analysis is complete
    return render_template('index.html')

#############################################################################################

def Numerical_distribution(df_cal, feature):
    """Distribution of numerical variable based on target variable"""
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    ax = sns.kdeplot(df_cal[feature], color='g', shade=True)
    title_str = "Original " + feature + " Distribution"
    plt.title(title_str)
    
    plt.subplot(2, 1, 2)
    ax = sns.kdeplot(df_cal.loc[df_cal['Churn'] == 1, feature], color='g', shade=True, label='Churn')
    ax = sns.kdeplot(df_cal.loc[df_cal['Churn'] == 0, feature], color='b', shade=True, label='No churn')
    title_str = feature + " Distribution: Churn vs No churn"
    plt.title(title_str)
    plt.show()

@app.route('/numerical_distribution')
def numerical_distribution_route():
    # Load the data from the CSV file
    data_file = 'files_for_training_model/telcom.csv'
    df_raw = pd.read_csv(data_file)

    # Preprocess the data if necessary (assuming Data_transformation_renaming is defined)
    df_cal = Data_transformation_renaming(df_raw)

    # Call the Numerical_distribution function with the desired feature names
    Numerical_distribution(df_cal, 'Tenure')
    Numerical_distribution(df_cal, 'MonthlyCharges')
    Numerical_distribution(df_cal, 'TotalCharges')
    # Render a template to display the analysis results
    return render_template('index.html')
#############################################################################################
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Rename columns and perform other transformations
    # Assuming Data_transformation_renaming function is defined
    df_cal = Data_transformation_renaming(df)
    
    # Normalize features
    df_cal['Tenure_norm'] = StandardScaler().fit_transform(df_cal[['Tenure']])
    df_cal['MonthlyCharges_norm'] = StandardScaler().fit_transform(df_cal[['MonthlyCharges']])
    return df_cal

# Route for displaying elbow curve
@app.route('/elbow_curve', methods=['GET'])
def elbow_curve_route():
    # Load data
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    
    # Preprocess data
    df_cal = preprocess_data(df_raw)
    
    # Create elbow curve
    df_churn = df_cal[df_cal['Churn'] == 1]
    Create_elbow_curve(df_churn[['Tenure_norm', 'MonthlyCharges_norm']])
    
    return render_template('index.html')

# Function to create elbow curve

def Create_elbow_curve(data):
    """ Display elbow curve for K-means algo for given data """
    df_kmeans_data = data

    k = range(1,10)
    kmeans = [KMeans(n_clusters=i) for i in k]

    score = [kmeans[i].fit(df_kmeans_data).score(df_kmeans_data)  for i in range(len(kmeans))]

    plt.figure(figsize=(10,6))
    plt.plot(k,score)
    plt.xlabel("Clusters")
    plt.ylabel("Score")
    plt.title("Elbow curve",fontsize=15)
    plt.show()

#############################################################################################
def Create_kmeans_cluster_graph(df_cal, data, n_clusters, x_title, y_title, chart_title):
    """ Display K-means cluster based on data """
    
    kmeans = KMeans(n_clusters=n_clusters # No of cluster in data
                    , random_state = random_state # Selecting same training data
                   ) 

    kmeans.fit(data)
    kmean_colors = [plotColor[c] for c in kmeans.labels_]


    fig = plt.figure(figsize=(12,8))
    plt.scatter(x= x_title + '_norm'
                , y= y_title + '_norm'
                , data=data 
                , color=kmean_colors # color of data points
                , alpha=0.25 # transparancy of data points
               )

    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.scatter(x=kmeans.cluster_centers_[:,0]
                , y=kmeans.cluster_centers_[:,1]
                , color='black'
                , marker='X' # Marker sign for data points
                , s=100 # marker size
               )
    
    plt.title(chart_title,fontsize=15)
    plt.show()
    
    return kmeans.fit_predict(df_cal[df_cal.Churn==1][[x_title+'_norm', y_title +'_norm']])


#############################################################################################
def Generate_bar_graph(x, y, x_title, y_title, chart_title,color=plotColor):
    """ Based on x and y value, generate bar graph """
    
    fig, ax = plt.subplots()
    ax.bar(range(len(x))
       , y
       , width = 0.75
       , color=color
        , alpha = 0.6) 

    # stopping alphabetical sorting of graph
    plt.xticks(range(len(x)),x)
    plt.title(chart_title, fontsize=14)
    plt.xlabel(x_title,fontsize=13)
    plt.ylabel(y_title,fontsize=13)
    plt.grid(b=False)
    plt.yticks(fontsize=0)
    plt.ylim(top=1)

    
    # Visible x - axis line
    for spine in plt.gca().spines.values():
        spine.set_visible(False) if spine.spine_type != 'bottom' else spine.set_visible(True)
    
    # Display label for each plot
    for i,v in (enumerate(y)):
        ax.text(i
                ,v+0.05
                ,str(round((v*100),2))+'%'
                ,fontsize=13
                ,ha='center')
    
    plt.show()

def Extract_highest_in_cluster(df_cal, df_cluster, feature, tenure_charges_cluster_df ):
    """ For each features, compare cluster's value with overall value 
    and find out highest distributed features for that cluster  """
    
    df = df_cal.copy()
    feature_churn_dist = df[(df['Churn']==1)][feature].value_counts(normalize=True).reset_index()
    feature_churn_dist.columns = [feature,'Percentage']
    feature_cluster_dist = df_cluster[feature].value_counts(normalize=True).to_frame()
    feature_cluster_dist.columns = ['Percentage']
    feature_cluster_dist = feature_cluster_dist.reset_index()
    feature_cluster_dist_new = feature_cluster_dist.copy()
    
    tenure_MonthlyCharges_df = df_cal[df_cal['Churn']==1].groupby(['Cluster',feature],as_index=False)['Tenure','MonthlyCharges'].mean()
    for i,cluster in enumerate(feature_cluster_dist_new['Cluster'].unique()):
        for i, label in enumerate(feature_churn_dist[feature].unique()):
            cluster_val = feature_cluster_dist_new[(feature_cluster_dist_new['Cluster']==cluster) & (feature_cluster_dist_new[feature]==label)]['Percentage']
            feature_val = feature_churn_dist[feature_churn_dist[feature] == label]['Percentage']
            
            if((len(feature_val.values) > 0) & (len(cluster_val.values) > 0)) :
                if((feature_val.values[0] < cluster_val.values[0])):
                    
                    tenure_charges_cluster_df = tenure_charges_cluster_df.append(pd.DataFrame({'Category':feature
                            , 'Label': ("Not have a "+ feature) if (df_cal[feature].dtypes == 'int64') & (label == 0) else (("Have a "+feature) if (df_cal[feature].dtypes == 'int64') & (label == 1) else label)
                            , 'Percentage': cluster_val.values[0]
                            , 'Cluster' : cluster
                            , 'Avg_Tenure': round(tenure_MonthlyCharges_df[(tenure_MonthlyCharges_df['Cluster']==cluster) & (tenure_MonthlyCharges_df[feature]==label) ]['Tenure'].values[0],2)
                            , 'Avg_MonthlyCharges': round(tenure_MonthlyCharges_df[(tenure_MonthlyCharges_df['Cluster']==cluster) & (tenure_MonthlyCharges_df[feature]==label) ]['MonthlyCharges'].values[0],2)
                            , 'Represent_in_graph': 0 if (label == 0) | (label == 'No') else 1
                            , 'Label_in_graph' :  feature if (df_cal[feature].dtypes == 'int64') else label
                           }
                        , index = [len(tenure_charges_cluster_df)])
                        )
                    
                    
    df_cal['Cluster'] = -1 # by default set Cluster to -1
    df_cal.loc[(df_cal.Churn==1),'Cluster'] = Create_kmeans_cluster_graph(df_cal
                            ,df_cal[df_cal.Churn==1][['Tenure_norm','MonthlyCharges_norm']]
                            ,3
                           ,'Tenure'
                           ,'MonthlyCharges'
                           ,"Tenure vs Monthlycharges : Churn customer cluster")

    df_cal['Cluster'].unique()               
    return tenure_charges_cluster_df
def process_clusters(df_cal):
    df_cal['Cluster'] = -1  # by default set Cluster to -1
    df_cal.loc[(df_cal.Churn == 1), 'Cluster'] = Create_kmeans_cluster_graph(df_cal,
                                                                             df_cal[df_cal.Churn == 1][['Tenure_norm', 'MonthlyCharges_norm']],
                                                                             3,
                                                                             'Tenure',
                                                                             'MonthlyCharges',
                                                                             "Tenure vs Monthlycharges : Churn customer cluster")

    df_cal['Cluster'].unique()
    tenure_charges_cluster_df = pd.DataFrame()
    df_cluster_gp = df_cal[df_cal['Churn'] == 1].groupby('Cluster')
    for feature in ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetServiceType',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'ContractType',
                    'PaperlessBilling', 'PaymentMethod', 'InternetService']:
        tenure_charges_cluster_df = Extract_highest_in_cluster(df_cal, df_cluster_gp, feature, tenure_charges_cluster_df)

    return tenure_charges_cluster_df

def Create_kmeans_cluster_with_label(data, cluster_visualize_gp, x_title, y_title, chart_title):
    """ Generate K-means cluster with labels """
    
    legend_list = []
    category_color = cluster_visualize_gp[['Category']].drop_duplicates().reset_index()
    annotations = []
    
    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(x= x_title 
                , y= y_title
                , data=data 
                , color=[plotColor[c] for c in data.Cluster] # color of data points
                , alpha=0.25 # transparancy of data points
                , s = 15
               )

    for i,txt in enumerate(cluster_visualize_gp['Label_in_graph']):
        annotations.append(ax.text(cluster_visualize_gp['Avg_Tenure'][i]
                                , cluster_visualize_gp['Avg_MonthlyCharges'][i]
                                , txt
                                , fontsize = 13
                                , weight="bold"))
        ax.scatter(x=cluster_visualize_gp['Avg_Tenure'][i]
            , y=cluster_visualize_gp['Avg_MonthlyCharges'][i]
            , color = plotColor[category_color[category_color['Category'] == cluster_visualize_gp['Category'][i]].index[0]]
            , label = cluster_visualize_gp['Label_in_graph'][i]
            , marker = markers[category_color[category_color['Category'] == cluster_visualize_gp['Category'][i]].index[0]]
            , s=120 # marker size
           )
    
    for key,i in enumerate(category_color.Category.values):
        legend_list.append(mlines.Line2D([]
                            , []
                            , linestyle= 'None'
                            , color = plotColor[key]      
                            , markersize = 10
                            , marker = markers[key]
                            , label= i))
    from adjustText import adjust_text
  
    adjust_text(annotations
                ,x=cluster_visualize_gp['Avg_Tenure']
                ,y=cluster_visualize_gp['Avg_MonthlyCharges'])
    
    plt.legend(handles=legend_list
              , loc = 'lower right')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(chart_title,fontsize=15)
    plt.show()


@app.route('/cluster_analysis')
def cluster_analysis_route():
    # Execute the provided code
    data_file = 'files_for_training_model/telcom.csv'
    extension = 'csv'
    data = Data()
    df_raw = data.Load_data(data_file, extension)
    
    # Preprocess data
    df_cal = preprocess_data(df_raw)
    tenure_charges_cluster_df = pd.DataFrame()

    # Process clusters
    df_cal['Cluster'] = -1
    df_cal.loc[(df_cal.Churn == 1), 'Cluster'] = Create_kmeans_cluster_graph(df_cal,
                                                                              df_cal[df_cal.Churn == 1][['Tenure_norm', 'MonthlyCharges_norm']],
                                                                              3,
                                                                              'Tenure',
                                                                              'MonthlyCharges',
                                                                              "Tenure vs Monthlycharges : Churn customer cluster")
    tenure_charges_cluster_df = process_clusters(df_cal)

    # Group by clusters
    df_cluster_gp = df_cal[df_cal['Churn'] == 1].groupby('Cluster')
    for feature in ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetServiceType',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'ContractType', 'PaperlessBilling', 'PaymentMethod', 'InternetService']:
        tenure_charges_cluster_df = Extract_highest_in_cluster(df_cal, df_cluster_gp, feature, tenure_charges_cluster_df)

    churn_distribution = df_cal[df_cal['Churn'] == 1].Cluster.value_counts(normalize=True).sort_index()

    # Generate bar graph
    Generate_bar_graph(x=churn_distribution.index,
                       y=churn_distribution.values,
                       x_title='Clusters',
                       y_title='Percentage',
                       chart_title='Cluster distribution',
                       color=plotColor)

    # Render an HTML template to display the visualization
    return render_template('index.html')


    # # Generate cluster visualizations
    # cluster_with_label_gp = tenure_charges_cluster_df[(tenure_charges_cluster_df['Represent_in_graph'] == 1) &
    #                                                   ((tenure_charges_cluster_df['Category'] == 'Gender') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'SeniorCitizen') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'Partner') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'Dependents'))]

    # Create_kmeans_cluster_with_label(df_cal[df_cal.Churn == 1][['Tenure', 'MonthlyCharges', 'Cluster']],
    #                                  cluster_with_label_gp.reset_index(),
    #                                  'Tenure',
    #                                  'MonthlyCharges',
    #                                  "Tenure vs Monthlycharges : Churn customer demographic cluster")

    # cluster_with_label_gp = tenure_charges_cluster_df[(tenure_charges_cluster_df['Represent_in_graph'] == 1) &
    #                                                   ((tenure_charges_cluster_df['Category'] == 'ContractType') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'PaperlessBilling') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'PaymentMethods'))]

    # Create_kmeans_cluster_with_label(df_cal[df_cal.Churn == 1][['Tenure', 'MonthlyCharges', 'Cluster']],
    #                                  cluster_with_label_gp.reset_index(),
    #                                  'Tenure',
    #                                  'MonthlyCharges',
    #                                  "Tenure vs Monthlycharges : Churn customer account based info")

    # cluster_with_label_gp = tenure_charges_cluster_df[(tenure_charges_cluster_df['Represent_in_graph'] == 1) &
    #                                                   ((tenure_charges_cluster_df['Category'] == 'StreamingTV') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'StreamingMovies') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'TechSupport') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'DeviceProtection') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'OnlineSupport') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'OnlineBackup') |
    #                                                    (tenure_charges_cluster_df['Category'] == 'InternetServiceType'))]

    # Create_kmeans_cluster_with_label(df_cal[df_cal.Churn == 1][['Tenure', 'MonthlyCharges', 'Cluster']],
    #                                  cluster_with_label_gp.reset_index(),
    #                                  'Tenure',
    #                                  'MonthlyCharges',
    #                                  "Tenure vs Monthlycharges : Churn customer usage based info")

    # Render an HTML template to display the visualization

#############################################################################################


@app.route('/retention_plan')
def retension():
    import base64

    with open("exist.png", 'rb') as f:
        image_data = f.read()

    encoded_image = base64.b64encode(image_data).decode('utf-8')

    image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Existing customer risk type distribution">'

    output = f"<h1>Existing customer risk type distribution</h1>{image_html}"

    return render_template('index.html', output=output)
#############################################################################################

@app.route('/retention_plan1')
def retension1():
    import base64

    with open("exist1.png", 'rb') as f:
        image_data = f.read()

    encoded_image = base64.b64encode(image_data).decode('utf-8')

    image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Existing customer risk type distribution">'

    output = f"<h1>Existing customers' Churn probability distribution</h1>{image_html}"

    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
