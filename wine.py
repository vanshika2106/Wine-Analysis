import pandas as pd
import numpy as np
import plotly.express as ex
import plotly.figure_factory as ff
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import joblib
warnings.filterwarnings('ignore')


# Preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,OrdinalEncoder,StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced


# Models
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree
from xgboost import XGBRegressor,XGBClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.linear_model import RidgeClassifier,RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,VotingClassifier,VotingRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential

#metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,plot_confusion_matrix,precision_score,recall_score
from sklearn.metrics import precision_recall_curve,roc_auc_score,roc_curve

# Model Evalution
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,train_test_split


import math
import warnings
import pickle

## ==============================================================================================================================================
## ================================================== Model Development +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## ==============================================================================================================================================

st.set_page_config(
     page_title="Wine Quality App",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "## We are building an interactive web app for building machine learning models"
     }
 )
st.sidebar.header("Hello There ")
menubar = st.sidebar.radio("",['Overview','Exploratory Data Analysis','Classification Model - Type','Classification Model - Quality','About Us'])

@st.cache(allow_output_mutation=True)
def inputdata():
    w = pd.read_csv('https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/wine.csv')
    w.columns = [i.replace(" ","_") for i in w.columns]
    w['discrete_quality'] = w.quality.apply(lambda x : 'Low' if x<=5 else ('High' if x>7 else 'Medium') )
    return w

@st.cache(allow_output_mutation=True)
def inputdata1():
    w = pd.read_csv('https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/wine.csv')
    w.columns = [i.replace(" ","_") for i in w.columns]
    w = w.dropna()
    w.drop_duplicates(subset=w.columns,keep='first',inplace=True)
    w['discrete_quality'] = w.quality.apply(lambda x : 'Low' if x<=5 else ('High' if x>7 else 'Medium') )

    return w


@st.cache(allow_output_mutation=True)
def cleaneddata():
    w = pd.read_csv('https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/wine.csv')
    w.columns = [i.replace(" ","_") for i in w.columns]
    w = w.dropna()
    w.drop_duplicates(subset=w.columns,keep='first',inplace=True)
    w['discrete_quality'] = w.quality.apply(lambda x : 'Low' if x<=5 else ('High' if x>7 else 'Medium') )
    sk = []
    for i in w.select_dtypes(exclude='object').columns:
        if abs(w[i].skew()) > 0.5:
            sk.append(i)
            w[i],skewness = stats.boxcox(w[i])

    for i in ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'pH', 'sulphates', 'alcohol']:
        IQR = w[i].describe()['75%'] - w[i].describe()['25%']
        lower = w[i].describe()['25%'] - (1.5*IQR)
        upper = w[i].describe()['75%'] + (1.5*IQR)
        if (((w[i]>upper) | (w[i]<lower)).sum() != 0):
            # '''Creating indicator columns for outliers 1 - Outlier and 0 - Not outlier'''
            # wine_outliers[i+"_outlier_indicator"] = ((wine_outliers[i]>upper) | (wine_outliers[i]<lower)).apply(lambda x : 1 if x else 0)

            # '''Replacing all outlier with NAN'''
            w[i].loc[((w[i]>upper) | (w[i]<lower))] = np.nan

    return w,sk


if menubar == 'Overview':
    st.markdown("<h1 style='text-align: center; color: Black;'>Wine Classifier - Type And Quality</h1>", unsafe_allow_html=True)
    wine = inputdata().copy()
    wine['discrete_quality'] = wine.quality.apply(lambda x : 'Low' if x<=5 else ('High' if x>7 else 'Medium') )
    st.dataframe(wine)
    st.write("[Download data as CSV ](https://raw.githubusercontent.com/VishnuNelapati/Zillow/main/wine.csv)")

    wine = wine.dropna()
    wine.drop_duplicates(subset=wine.columns,keep='first',inplace=True)

    st.markdown("<h4 style='text-align: left; color: Black;'>Description of the data and columns </h4>", unsafe_allow_html=True)


    # def report():
    #     profile = ProfileReport(wine,explorative = True)
    #     return profile
    #
    # st_profile_report(report())


    text = ['The predominant fixed acids found in wines are tartaric, malic, citric, and succinic. Their respective levels found in wine can vary greatly but in general one would expect to see 1,000 to 4,000 mg/L tartaric acid, 0 to 8,000 mg/L malic acid, 0 to 500 mg/L citric acid, and 500 to 2,000 mg/L succinic acid.',
    ' In general, per the CFR: "The maximum volatile acidity, calculated as acetic acid and exclusive of sulfur dioxide, is 0.14 g/100 mL for red wine and 0.12 g/100 mL for white wines." This is equivalent to 1.4 and 1.2 g/L acetic acid for red and white wines, respectively.',
    'Citric acid is often added to wines to increase acidity, complement a specific flavor or prevent ferric hazes. It can be added to finished wines to increase acidity and give a “fresh” flavor.Since bacteria use citric acid in their metabolism, it may increase the growth of unwanted microbes.',
    'Residual Sugar (or RS) is from natural grape sugars leftover in a wine after the alcoholic fermentation finishes. It is measured in grams per liter. So for example, a wine with 10 grams per liter of residual sugar has 1% sweetness or a total of ~1.8 carbohydrates per serving (5 ounces / 150 ml).',
    'Wine contains from 2 to 4 g L–1 of salts of mineral acids, along with some organic acids, and they may have a key role on a potential salty taste of a wine, with chlorides being a major contributor to saltiness (Walker et al., 2003; Maltman, 2013).',
    'Total Sulfur Dioxide (TSO2) is the portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugars.',
    'Total Sulfur Dioxide (TSO2) is the portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugars.',
    'The typical density or specific gravity of the must (the term we give to wine before we add or pitch the yeast) is generally between 1.080 and 1.090. This essentially means your wine is 8-9% more dense than water.',
    'Typically, the pH level of a wine ranges from 3 to 4. Red wines with higher acidity are more likely to be a bright ruby color, as the lower pH gives them a red hue.',
    'Wine sulfites are naturally occurring at low levels in all wines, and are one of the thousands of chemical by-products created during the fermentation process. However, sulfites are also added by the winemaker to preserve and protect the wine from bacteria and yeast-laden invasions',
    'Wine can have anywhere between 5% and 23% ABV. The average alcohol content of wine is about 12%. This amount varies depending on the variety of wine, as well as the winemaker and their desired ABV.']

    for j,i in  enumerate(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']):

       with st.expander(i,True):
           st.markdown(f"<h5 style='text-align: center; color: Black;'>{i.upper()}</h5>", unsafe_allow_html=True)
           c2,c3,c4 = st.columns((1,0.05,1))

           st.markdown(text[j])

           with c4:
               fig1 = ex.histogram(wine, x=i, color="discrete_quality", marginal="box",
                      hover_data=wine.columns)
               fig1.update_layout(title =f"Distribution of {i} by Quality of Wine",title_x = 0.5,height = 400,width = 650)
               st.plotly_chart(fig1,se_container_width=True)

           with c2:
               fig = ex.histogram(wine, x=i, color="Type", marginal="box",
                      hover_data=wine.columns,color_discrete_sequence = ("grey",'orangered'))
               fig.update_layout(title =f"Distribution of {i} by Type of Wine",title_x = 0.5,height = 400,width = 650)
               st.plotly_chart(fig)


if menubar == 'Exploratory Data Analysis':

    wine1 = inputdata1().copy()

    st.markdown("<h1 style='text-align: center; color: Black;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    st.subheader("Scatter Plot Matrix")

    exp1,exp2,exp3 = st.columns((0.4,0.05,1))
    with exp1:

        st.write("Choose the Columns")

        f1 = st.selectbox('Feature 1',wine1.columns,0)

        f2 = st.selectbox("Feature 2",wine1.columns,1)

        f3 = st.selectbox('Feature 3',wine1.columns,2)

        f4 = st.selectbox("Color",['Type','quality','discrete_quality'])

        f5 = st.selectbox("Dimension",['2D Matrix','3D'],1)

    with exp3:

        if f4 == 'Type':
            cds = ('grey','orangered')
        else:
            cds =('#636efa','#e45744','#06cb94')

        if f5 == '3D':

            fig = ex.scatter_3d(wine1,x = f1,y = f2,z = f3,color = f4,opacity = 0.7,color_discrete_sequence = cds)

            fig.update_layout(
            title=f'3D Scatter Plot - By {f4}',
            dragmode='select',
            width=1000,
            height=600,
            hovermode='closest',
            title_x = 0.5,
        )

            st.plotly_chart(fig)
        else:
            fig = ex.scatter_matrix(wine1,
            dimensions=[f1,f2,f3],opacity = 0.7,
            color=f4,color_discrete_sequence = cds)

            fig.update_layout(
            title='Scatter Matrix Plot - Wine Data Set',
            dragmode='select',
            width=1000,
            height=600,
            hovermode='closest',title_x = 0.5,
        )
            st.plotly_chart(fig)



    st.write("---")
    h1,h2,h3 = st.columns((0.5,0.05,1))

    with h1:
        st.subheader("Heat Map")

        st.markdown(''' - Correlation gives the statistical relationship between two entities and helps us to know how the two variables move in relation to one another. This relation can be easily identified by using heat maps which represent the strength of the relationship using different colors.
- From the Correlation Heat map , we can observe from the below plot there is high correlation between free_sulpur dioxide and total_sulpher dioxide. Highly correlated data leads to multicollinearity or data leakage. If there is multicollinearity then we cannot determine regression coefficients uniquely and face problems in predicting the model.
- While building machine learning models it is important that we remove or transform all highly correlated features from the input data-set to avoid overfitting and to increase the model performance. PCA is one such method which helps in dealing with multicollinearity, which we will be applying before building our models.
        ''')


    with h3:
        corr = wine1.corr()

        # fig = ex.imshow(corr,color_continuous_scale='gnbu')
        # fig.update_layout(
        # title='Correlation Heat Map',
        # width=800,
        # height=600,
        # hovermode='closest',title_x = 0.5)
        #
        # st.plotly_chart(fig)

        fig = ff.create_annotated_heatmap(z = np.round(wine1.corr().values,2),x = list(wine1.select_dtypes(exclude = 'object').columns),
        y = list(wine1.select_dtypes(exclude = 'object').columns),colorscale='viridis')
        fig.update_layout(
        title='Correlation Heat Map',
        width=900,
        height=600,
        hovermode='closest',title_x = 0.5,title_y = 0.05)
        st.plotly_chart(fig)


    st.subheader("Skewness of the data")

    st.markdown('''The rule of thumb for skewness :
- If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
- If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.
- If the skewness is less than -1 or greater than 1, the data are highly skewed.''')
    c1,c2,c3 = st.columns((1,0.05,1))

    with c1:
        c = 0
        skewed_columns = []
        fig1 = plt.figure(figsize=(20,20))
        for i in wine1.select_dtypes(exclude='object').columns:
            c =c +1
            plt.subplot(4,3,c)
            sns.distplot(x = wine1[i])
            if abs(wine1[i].skew()) > 0.5:
                skewed_columns.append(i)
            plt.xlabel(f"Skewness: {round(wine1[i].skew(),3)}",fontsize = 15)
            plt.ylabel(i,fontsize = 15)
        plt.suptitle("Original Data",color = 'red',fontsize = 26,y = 0.92)
        st.pyplot(fig1)

    st.cache(allow_output_mutation = True)
    def skewdata(skewed_columns):
        for i in skewed_columns:
            wine1[i],skewness = stats.boxcox(wine1[i])

        return wine1

    with c3:

        wine1 = skewdata(skewed_columns).copy()
        c = 0
        fig2 = plt.figure(figsize=(20,20))
        for i in wine1.select_dtypes(exclude='object').columns:
            c =c +1
            plt.subplot(4,3,c)
            sns.distplot(x = wine1[i])
            plt.xlabel(f"Skewness: {round(wine1[i].skew(),3)}",fontsize = 15)
            plt.ylabel(i,fontsize = 15)
        plt.suptitle("Non-Skewed Data",color = 'red',fontsize = 26,y = 0.92)
        st.pyplot(fig2)
    st.write("---")

    st.subheader("Visualizing The outliers")

    st.markdown('''It is always suggested to remove the outliers from the dataset as outliers will always provide biased outputs and the analysis outcomes may be misleading. In the above set of graphs, let us consider fixed_acidity and Type graphs – all the points lying above the upper whisker and below the lower whisker are the outliers and these values need to be replaced with the null values. Similarly, we need to find the outliers for other graphed columns and replace them.
Later, all the null values can be filled with imputed values like mean or median or mode based on the requirement.
''')

    c1,c2,c3 = st.columns((1,0.05,1))

    with c1:
        c = 0
        fig3 = plt.figure(figsize=(20,15))
        for i in wine1.select_dtypes(exclude='object').columns:
            c =c +1
            plt.subplot(3,4,c)
            sns.boxplot(data=wine1,y=i,x = 'Type',palette=('#FFFFFF','red'))
            plt.ylabel(i)
        plt.suptitle("Box Plot Against Type of Wine",color = 'red',fontsize = 26,y = 0.92)
        st.pyplot(fig3)

    with c3:
        c = 0
        fig3 = plt.figure(figsize=(20,15))
        for i in wine1.select_dtypes(exclude='object').columns:
            c =c +1
            plt.subplot(3,4,c)
            sns.boxplot(data=wine1,y=i,x = 'discrete_quality')
            plt.ylabel(i)
        plt.suptitle("Box Plot Against Quality of Wine",color = 'red',fontsize = 26,y = 0.92)
        st.pyplot(fig3)

    st.write("---")
#==================================================================================================================================================
#==========================================Classification MOdel - Type ===========================================================================
#==================================================================================================================================================

if menubar == 'Classification Model - Type':

    st.markdown("<h1 style='text-align: center; color: Black;'>Classification - Wine Type</h1>", unsafe_allow_html=True)

    wine2 = cleaneddata()[0].copy()

    y = wine2[['Type']]
    X = wine2.drop(['Type','discrete_quality'],axis = 1)
    y_discrete = y.copy()
    ord_encoder =OrdinalEncoder()
    y['Type'] = ord_encoder.fit_transform(y)
    targets = list(ord_encoder.categories_[0])

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

    #Numeric columns
    num_cols = X_train.select_dtypes(exclude = 'object').columns
    cat_cols = X_train.select_dtypes(include= 'object').columns

    #Numeric Preprocessing
    num_imputer = KNNImputer(n_neighbors=5)


    #Categorical Preprocessing
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_encoder = OneHotEncoder(drop='first')

    #Defining Categorical Pipeline
    cat_pipeline = Pipeline(steps=[('cat_imputer',cat_imputer),
                                  ('cat_encoding',cat_encoder)])

    #Defininh column Transformer
    column_tranformation = ColumnTransformer(transformers=[('numeric_imputer',num_imputer,num_cols),
                                            ('categorical_transformation',cat_pipeline,cat_cols)],
                                            remainder='passthrough',verbose=False)

    scaler = MinMaxScaler()

    # PCA
    comp = len(num_cols)+sum([len(X_train[i].unique()) for i in cat_cols])-len(cat_cols)
    pca = PCA(n_components=comp)

    # SMOTE
    over_samp = SMOTE(sampling_strategy='minority',random_state=1)


    # =======================================================================================================================================================
    # ========================================================= Why PCA =====================================================================================
    # =======================================================================================================================================================


    bal,bi,imbal = st.columns((1.4,0.1,1))

    with bal:
        with st.container():
            pass
            st.markdown(''' ## What is a Pipeline ?

**Pipelines** sequentially apply a list of transforms and a final estimator.Intermediate steps of the pipeline must be transforms, that is, they must implement `fit` and `transform` methods. The final estimator only needs to implement `fit`.
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a `'__'`, as in the example below. A step's estimator may be replaced entirely by setting the parameter with its name
to another estimator, or a transformer removed by setting it to `'passthrough'` or `None`.
Most people hack together models without pipelines, but pipelines have some important benefits. Those include:
- **Cleaner Code:** Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step. This will help in avoiding data leakage.
- **Fewer Bugs:** There are fewer opportunities to misapply a step or forget a preprocessing step.
- **Easier to Productionize:** It can be surprisingly hard to transition a model from a prototype to something deployable at scale.
- **More Options for Model Validation:** All parameters can be passed at once to the pipeline steps like PCA , Model Building , Scaling etc. .

Pipelines are valuable for cleaning up machine learning code and avoiding errors, and are especially useful for workflows with sophisticated data preprocessing.
            ''')

    with imbal:
        with st.container():
            st.write("")
            st.write("")

            imb = st.radio("",['Balanced Pipeline','Imbalance Pipeline'],0)

            if imb == 'Imbalance Pipeline':
                t = '<p style="color:red; font-size: 28px;">Pipeline For Imbalanced Classes</p>'
                st.markdown(t, unsafe_allow_html=True)
                st.markdown('''

        <div class="output_subarea output_html rendered_html output_result" dir="auto"><style>#sk-965986a2-f7e6-499a-b39e-ff3be569b002 {color: black;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 pre{padding: 0;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable {background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator:hover {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-item {z-index: 1;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:only-child::after {width: 0;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-container {display: inline-block;position: relative;}</style><div id="sk-965986a2-f7e6-499a-b39e-ff3be569b002" class"sk-top-container"=""><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d9aa7a3a-615c-49c2-a4e1-64615f94702a" type="checkbox"><label class="sk-toggleable__label" for="d9aa7a3a-615c-49c2-a4e1-64615f94702a">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntransformer',
                         ColumnTransformer(remainder='passthrough',
                                           transformers=[('Numeric',
                                                          KNNImputer(),
                                                          Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')),
                                                         ('Categorical',
                                                          Pipeline(steps=[('cat_imputer',
                                                                           SimpleImputer(strategy='most_frequent')),
                                                                          ('cat_encoding',
                                                                           OneHotEncoder(drop='first'))]),
                                                          Index([], dtype='object'))])),
                        ('minmaxscaler', MinMaxScaler()), ('pca', PCA(n_components=12)),
                        ('smote', SMOTE(random_state=1, sampling_strategy='minority')),
                        ('Model', Model())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dfb7081c-f09b-42e1-b47e-bb554b2e37c9" type="checkbox"><label class="sk-toggleable__label" for="dfb7081c-f09b-42e1-b47e-bb554b2e37c9">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder='passthrough',
                          transformers=[('Numeric', KNNImputer(),
                                         Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')),
                                        ('Categorical',
                                         Pipeline(steps=[('cat_imputer',
                                                          SimpleImputer(strategy='most_frequent')),
                                                         ('cat_encoding',
                                                          OneHotEncoder(drop='first'))]),
                                         Index([], dtype='object'))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="18917d44-ba2a-40e9-9129-373096147cf7" type="checkbox"><label class="sk-toggleable__label" for="18917d44-ba2a-40e9-9129-373096147cf7">Numeric</label><div class="sk-toggleable__content"><pre>Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9d30486d-3aec-4eb3-8e5c-259f262596aa" type="checkbox"><label class="sk-toggleable__label" for="9d30486d-3aec-4eb3-8e5c-259f262596aa">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="665c7bc8-ae80-4a1f-a37b-3937b3d60589" type="checkbox"><label class="sk-toggleable__label" for="665c7bc8-ae80-4a1f-a37b-3937b3d60589">Categorical</label><div class="sk-toggleable__content"><pre>Index([], dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c95c2466-e267-4b13-bfd9-e4a2ba1b33dd" type="checkbox"><label class="sk-toggleable__label" for="c95c2466-e267-4b13-bfd9-e4a2ba1b33dd">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9ffb8bde-d335-40d5-ae5d-a12c9f02b2ce" type="checkbox"><label class="sk-toggleable__label" for="9ffb8bde-d335-40d5-ae5d-a12c9f02b2ce">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop='first')</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ff39a13f-c1ae-4c28-8e90-246b16537e29" type="checkbox"><label class="sk-toggleable__label" for="ff39a13f-c1ae-4c28-8e90-246b16537e29">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2051673c-72eb-4b18-8985-82361a467d00" type="checkbox"><label class="sk-toggleable__label" for="2051673c-72eb-4b18-8985-82361a467d00">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0168020f-6535-42fb-a9b0-9c9429487023" type="checkbox"><label class="sk-toggleable__label" for="0168020f-6535-42fb-a9b0-9c9429487023">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1b5dfe2c-e4b1-413c-9252-e9f2699f2249" type="checkbox"><label class="sk-toggleable__label" for="1b5dfe2c-e4b1-413c-9252-e9f2699f2249">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=12)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a9116615-95de-4e33-a12a-0c428be63466" type="checkbox"><label class="sk-toggleable__label" for="a9116615-95de-4e33-a12a-0c428be63466">SMOTE</label><div class="sk-toggleable__content"><pre>SMOTE(random_state=1, sampling_strategy='minority')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="aa6e972a-f359-467e-b448-98ecf6528066" type="checkbox"><label class="sk-toggleable__label" for="aa6e972a-f359-467e-b448-98ecf6528066">Model</label><div class="sk-toggleable__content"><pre>Model()</pre></div></div></div></div></div></div></div></div>

            ''',unsafe_allow_html=True)

            else:
                t = '<p style="color:red;font-size: 28px;">Pipeline For Balanced Classes</p>'
                st.markdown(t, unsafe_allow_html=True)
                st.markdown('''
            <div class="output_subarea output_html rendered_html output_result" dir="auto"><style>#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e {color: black;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e pre{padding: 0;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable {background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator:hover {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-item {z-index: 1;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:only-child::after {width: 0;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-container {display: inline-block;position: relative;}</style><div id="sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e" class"sk-top-container"=""><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3fc52e7f-cef0-4022-8167-d9180200b176" type="checkbox"><label class="sk-toggleable__label" for="3fc52e7f-cef0-4022-8167-d9180200b176">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntranformation',
                             ColumnTransformer(remainder='passthrough',
                                               transformers=[('Numeric',
                                                              KNNImputer(),
                                                              Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')),
                                                             ('Categorical',
                                                              Pipeline(steps=[('cat_imputer',
                                                                               SimpleImputer(strategy='most_frequent')),
                                                                              ('cat_encoding',
                                                                               OneHotEncoder(drop='first'))]),
                                                              Index([], dtype='object'))])),
                            ('scaler', MinMaxScaler()), ('pca', PCA(n_components=12)),
                            ('model',
                             Model())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a52cc582-3160-4269-891b-ab7d46d7181c" type="checkbox"><label class="sk-toggleable__label" for="a52cc582-3160-4269-891b-ab7d46d7181c">columntranformation: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder='passthrough',
                              transformers=[('Numeric', KNNImputer(),
                                             Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')),
                                            ('Categorical',
                                             Pipeline(steps=[('cat_imputer',
                                                              SimpleImputer(strategy='most_frequent')),
                                                             ('cat_encoding',
                                                              OneHotEncoder(drop='first'))]),
                                             Index([], dtype='object'))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ecfe53b2-31ee-4c5f-9321-35e553ad7e70" type="checkbox"><label class="sk-toggleable__label" for="ecfe53b2-31ee-4c5f-9321-35e553ad7e70">Numeric</label><div class="sk-toggleable__content"><pre>Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0f37bc6c-3f7b-4142-9a6d-41392201012c" type="checkbox"><label class="sk-toggleable__label" for="0f37bc6c-3f7b-4142-9a6d-41392201012c">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="aada903c-d6d6-4a2e-bfb7-8e27879189e7" type="checkbox"><label class="sk-toggleable__label" for="aada903c-d6d6-4a2e-bfb7-8e27879189e7">Categorical</label><div class="sk-toggleable__content"><pre>Index([], dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5e6d1826-feb8-4fe9-a7d1-e25ef25d769e" type="checkbox"><label class="sk-toggleable__label" for="5e6d1826-feb8-4fe9-a7d1-e25ef25d769e">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1f87e7dd-bd06-49c0-b47a-7ecede5f8cc1" type="checkbox"><label class="sk-toggleable__label" for="1f87e7dd-bd06-49c0-b47a-7ecede5f8cc1">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop='first')</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="90e72020-58ae-44c4-87fe-cc058ecbad63" type="checkbox"><label class="sk-toggleable__label" for="90e72020-58ae-44c4-87fe-cc058ecbad63">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e6577c91-8c2f-418b-b805-41a59e80d959" type="checkbox"><label class="sk-toggleable__label" for="e6577c91-8c2f-418b-b805-41a59e80d959">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3d6e01d2-0948-4807-8c80-c916aceef3dd" type="checkbox"><label class="sk-toggleable__label" for="3d6e01d2-0948-4807-8c80-c916aceef3dd">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a4c4ddd8-eb12-42e1-a17a-26a3b4f1ff9b" type="checkbox"><label class="sk-toggleable__label" for="a4c4ddd8-eb12-42e1-a17a-26a3b4f1ff9b">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=12)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d69272fd-f66a-4236-a46f-15801bb9c854" type="checkbox"><label class="sk-toggleable__label" for="d69272fd-f66a-4236-a46f-15801bb9c854">model</label><div class="sk-toggleable__content"><pre>Model()</pre></div></div></div></div></div></div></div></div>
                ''', unsafe_allow_html=True)

    st.write("---")

    c0,c1,c2,c3 = st.columns((0.02,0.60,0.02,1))
    o0,o1,o2,o3 = st.columns((0.9,0.55,0.55,0.15))

    with o1:
        scale = st.checkbox('PCA without Scaling',key = 's')
        exp1 = st.expander("Why Scaling ? ")
        with exp1:

            st.caption('''- Many machine learning algorithms perform better or converge faster when features are on a relatively similar scale and/or close to normally distributed.

- Machine learning algorithm just sees number , if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort.''')

    with o2:
        smote = st.checkbox('Oversample Minority Class',key = 'smote')
        exp2 = st.expander("Why OverSample Minority class ? ")

        with exp2:

            st.caption('''- Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce errors.
- However, if the data set in imbalance then In such cases, you get a **pretty high accuracy just by predicting the majority class, but you fail to capture the minority class**, which is most often the point of creating the model in the first place.
- **SMOTE** is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.''')

    with o3:

        dimension = st.select_slider("Dimension:",['3D','2D'],"2D",key = 1)

        if not scale:
            pca_pipeline = Pipeline(steps = [("ct",column_tranformation),('sclaer',scaler),('pca',pca)])
            pc_df = pd.DataFrame(pca_pipeline.fit_transform(X_train),columns = ['PC'+str(i) for i in range(1,13)],index = X_train.index)
            pc_df = pc_df.iloc[:,0:3]
            labels = pd.DataFrame(ord_encoder.inverse_transform(y_train),index = y_train.index,columns = ['Type'])
            full_df = pd.concat([pc_df,labels],axis = 1)

            if smote:
                pca_pipeline = make_pipeline(column_tranformation,scaler,pca,over_samp)
                ox,oy = pca_pipeline.fit_resample(X_train,y_train)
                pc_df = pd.DataFrame(ox,columns = ['PC'+str(i) for i in range(1,13)])
                pc_df = pc_df.iloc[:,0:3]
                labels = pd.DataFrame(ord_encoder.inverse_transform(oy),columns = ['Type'])
                full_df = pd.concat([pc_df,labels],axis = 1)

        else:

            pca_pipeline = Pipeline(steps = [("ct",column_tranformation),('pca',pca)])
            pc_df = pd.DataFrame(pca_pipeline.fit_transform(X_train),columns = ['PC'+str(i) for i in range(1,13)],index = X_train.index)
            pc_df = pc_df.iloc[:,0:3]
            labels = pd.DataFrame(ord_encoder.inverse_transform(y_train),index = y_train.index,columns = ['Type'])
            full_df = pd.concat([pc_df,labels],axis = 1)

            if smote:
                pca_pipeline = make_pipeline(column_tranformation,pca,over_samp)
                ox,oy = pca_pipeline.fit_resample(X_train,y_train)
                pc_df = pd.DataFrame(ox,columns = ['PC'+str(i) for i in range(1,13)])
                pc_df = pc_df.iloc[:,0:3]
                labels = pd.DataFrame(ord_encoder.inverse_transform(oy),columns = ['Type'])
                full_df = pd.concat([pc_df,labels],axis = 1)

            # if smote:
            #     pca_pipeline = make_pipeline(column_tranformation,pca,over_samp)
            #     ox,oy = pca_pipeline.fit_resample(X_train,y_train)
            #     pc_df = pd.DataFrame(ox,columns = ['PC'+str(i) for i in range(1,13)])
            #     pc_df = pc_df.iloc[:,0:3]
            #     full_df = pd.concat([pc_df,labels],axis = 1)


    with c1:
        st.markdown("## Principle Component Analysis")

        st.markdown('''Principal component analysis (PCA) is one of a family of techniques for taking high-dimensional data, and using the dependencies between the variables to represent it in a more tractable, lower-dimensional basis, without losing too much information.
It is Statistical procedure to convert observations of possibly correlated variables into 'Principal Components' that are:
- Independent of each other (Uncorrelated)
- That Captures maximum information in first few components

- **UseCases** :
    - Dimensionality reduction
    - Data visualization and Exploratory Data Analysis
    - Create uncorrelated features/variables that can be an input to a prediction model
    - Uncovering latent variables/themes/concepts
    - Noise reduction in the dataset
''')

    with o0:
        fig = ex.line(y = pca_pipeline.named_steps.pca.explained_variance_ratio_,x = ['PC'+str(i) for i in range(1,13,1)],labels = {'x':'Components','y':'Explained Variance (%)'})
        fig.update_layout(showlegend=False,width = 600,height = 400,title = "Scree Plot",title_x = 0.5)
        st.plotly_chart(fig)

        with st.expander("Scree Plot"):
            st.caption('''**Screen Plot** helps in identifying the components required for building a model.After a certain point the variance tends to decreese very minimal which indicates the number of components.This graph is generally defined as Elbow Plot or Scree Plot''')

    # fig = ex.line(pca_pipeline.named_steps.pca.explained_variance_ratio_,labels = {'index':'Components','value':'Cumulative Explained Variance'})
    # fig.update_layout(showlegend=False)
    # st.plotly_chart(fig)

    with c3:

        st.write(" ")
        if dimension == '3D':

            fig = ex.scatter_3d(full_df, x='PC1', y='PC2', z='PC3',
                      color='Type',opacity=0.7,color_discrete_sequence = ('orangered','gray'))

            fig.update_layout(width = 600,height = 600,title = "Visualizing the Type of wine - 3D",title_x = 0.5,title_y = 0.95)

            st.plotly_chart(fig,use_container_width=True)

        elif dimension == '2D':

            fig = ex.scatter(full_df, x='PC1', y='PC2',
                      color='Type',opacity=0.7,color_discrete_sequence = ('orangered','gray'))

            fig.update_layout(width = 600,height = 600,title = "Visualizing the Type of wine - 2D",title_x = 0.5,title_y = 0.95)

            st.plotly_chart(fig,use_container_width=True)

    st.write("---")


    #=============================================================================================================================================
    #========================================================Model Building ======================================================================
    #=============================================================================================================================================


    st.markdown(" ## Models Evaluation")

    st.write('Test Data')

    ind = np.random.randint(10,len(X_test)-10)
    input = X_test.iloc[ind:ind+1,:]
    # st.write(input)
    #
    # st.write(input.index[0])

    dis = inputdata1()[inputdata1().index == input.index[0]]
    st.dataframe(dis[X_train.columns])

    # st.selectbox("Choose the model",['Logistic Regression','Decision Tree Classifier',
    # 'Random Forrest Classifier','XGBoost Classifier','Neural Network','Adaboost Classifier','Gradient Boost Classifier'])

    pred = st.button("Predict Type")

    # loaded_model = joblib.load('decisiontreeclassifier-Imbalance.pkl')
    # model_output = "Red" if loaded_model.predict(input)==0 else 'White'
    # si = 1 if model_output ==  np.array(dis.Type)[0] else -1

    m1,m2,m3,m4,m5,m6,m7 = st.columns((1,1,1,1,1,1,1))

    if pred:
        with m1:
            st.metric('Actual Value',np.array(dis.Type)[0],)
        with m2:
            loaded_model = joblib.load('logisticregression-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('Logistic Regression',model_output,si)
        with m3:
            loaded_model = joblib.load('decisiontreeclassifier-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('Decision Tree Classifier',model_output,si)
        with m4:
            loaded_model = joblib.load('randomforestclassifier-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('Random Forest Classifier',model_output,si)
        with m5:
            loaded_model = joblib.load('xgbclassifier-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('XGBoost Classifier',model_output,si)
        with m6:
            loaded_model = joblib.load('adaboostclassifier-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('AdaBoost Classifier',model_output,si)
        with m7:
            loaded_model = joblib.load('gradientboostingclassifier-Imbalance.pkl')
            model_output = "Red" if loaded_model.predict(input)==0 else 'White'
            si = 1 if model_output ==  np.array(dis.Type)[0] else -1
            st.metric('Gradient Boost Classifier',model_output,si)

    else:
        st.metric('Actual Value',np.array(dis.Type)[0],)



#==================================================================================================================================================
#==========================================Classification MOdel - Quality ===========================================================================
#==================================================================================================================================================


if menubar == 'Classification Model - Quality':

    st.markdown("<h1 style='text-align: center; color: Black;'>Classification - Quality of Wine</h1>", unsafe_allow_html=True)

    wine3 = cleaneddata()[0].copy()

    y = wine3[['discrete_quality']]
    X = wine3.drop(['quality','discrete_quality'],axis = 1)
    ord_encoder =OrdinalEncoder()
    y['discrete_quality'] = ord_encoder.fit_transform(y)
    targets = list(ord_encoder.categories_[0])

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=1)

    #Numeric columns
    num_cols = X_train.select_dtypes(exclude = 'object').columns
    cat_cols = X_train.select_dtypes(include= 'object').columns

    #Numeric Preprocessing
    num_imputer = KNNImputer(n_neighbors=5)


    #Categorical Preprocessing
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_encoder = OneHotEncoder(drop='first')

    #Defining Categorical Pipeline
    cat_pipeline = Pipeline(steps=[('cat_imputer',cat_imputer),
                                  ('cat_encoding',cat_encoder)])

    #Defininh column Transformer
    column_tranformation = ColumnTransformer(transformers=[('numeric_imputer',num_imputer,num_cols),
                                            ('categorical_transformation',cat_pipeline,cat_cols)],
                                            remainder='passthrough',verbose=False)


    scaler = MinMaxScaler()

    # PCA
    comp = len(num_cols)+sum([len(X_train[i].unique()) for i in cat_cols])-len(cat_cols)
    pca = PCA(n_components=comp)

    # SMOTE
    over_samp = SMOTE(sampling_strategy='minority',random_state=1)

    def model_metrics_mul(grid,model_type,X_test = X_test,y_test = y_test,targets = targets):



        st.markdown("**Best Parameters:**")
        st.write(grid.best_params_)

        y_pred = grid.predict(X_test)
        y_pred_prob = grid.predict_proba(X_test)

        features = X_test.columns

        st.write("Performance Metrics:")
        st.markdown(f"Accuracy - {round(accuracy_score(y_test,y_pred),2)}")
    #     print("Precision",round(precision_score(y_test,y_pred),2))
    #     print("Recall",round(recall_score(y_test,y_pred),2))


        # st.write("\n","Confusion Matrix :")
        cm = confusion_matrix(y_test,y_pred)
        total_classes = len(targets)
        codes = [total_classes*[0], list(range(total_classes))]
        cm = confusion_matrix(y_test, y_pred)
        cm_frame = pd.DataFrame(data=cm,
                                columns=pd.MultiIndex(levels=[['Predicted'], targets], codes=codes),
                                index=pd.MultiIndex(levels=[['Actual'], targets], codes=codes))

        print("\n","Classification Report :")
        print("-"*50)
        st.markdown(classification_report(y_test,y_pred,target_names=targets))
        # st.write(cm_frame)
        m0,m1 = st.columns((1,1))
        m2,m3 = st.columns((1,1))

        with m0:

            fig = plt.figure(figsize = (4,4))
            sns.heatmap(cm_frame,cbar=False,annot=True,fmt="d",xticklabels=True,yticklabels=True)
            plt.xticks([0.5,1.5,2.5],labels=targets)
            plt.yticks([0.5,1.5,2.5],labels=targets)
            plt.title('Confusion matrix')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.show()
            st.pyplot(fig)
        with m1:

            fpr = {}
            tpr = {}
            thresh ={}

            for i in range(len(targets)):
                fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_prob[:,i], pos_label=i)

            # plotting
            fig = plt.figure(figsize = (4,4))
            plt.plot(fpr[0], tpr[0],color='red', label='High vs Rest')
            plt.plot(fpr[1], tpr[1],color='green', label='Low vs Rest')
            plt.plot(fpr[2], tpr[2],color='blue', label='Medium vs Rest')
            plt.plot([(0,0),(1,1)],linestyle = "--",color = "black")
            plt.title('ROC curve')
            plt.legend((f"ROC ['High vs Rest']",
                        f"ROC ['Low vs Rest']",
                        f"ROC ['Medium vs Rest']",
                        "Random"))
            plt.xlabel('False Positive Rate')
            plt.ylabel("True Positive Rate")
            plt.show()
            st.pyplot(fig)

        with m2:

            pre = {}
            rec = {}
            thresh1 ={}

            for i in range(len(targets)):
                pre[i], rec[i], thresh1[i] = precision_recall_curve(y_test, y_pred_prob[:,i], pos_label=i)

            # plotting
            fig = plt.figure(figsize = (4,4))
            plt.plot(pre[0], rec[0],color='red', label='High vs Rest')
            plt.plot(pre[1], rec[1],color='green', label='Low vs Rest')
            plt.plot(pre[2], rec[2],color='blue', label='Medium vs Rest')
            plt.plot([(0,0),(1,1)],linestyle = "--",color = "black")
            plt.title('Precision - Recall curve')
            plt.legend((f"PR ['High vs Rest']",
                        f"PR ['Low vs Rest']",
                        f"PR ['Medium vs Rest']",
                        "Random"))
            plt.xlabel('Recall')
            plt.ylabel("Precision")
            plt.show()
            st.pyplot(fig)

        try:
            # Variable importance
            fig = plt.figure(figsize = (4,4))
            importance = pd.DataFrame({'features': features,'importance': grid.best_estimator_.named_steps[model_type].feature_importances_})
            importance.index=importance.features
            importance.sort_values(by='importance', ascending=True).plot.barh()
            plt.legend(loc='lower right')
            plt.title("Feature Importance Plot")
            plt.show()
            st.pyplot(fig)
        except:
            fig = plt.figure(figsize = (4,4))
            importance = pd.DataFrame(grid.best_estimator_.named_steps[model_type].coef_).T
            importance.columns = targets
            importance.index = features
            fig = importance.sort_values(by=targets, ascending=True).plot.barh()
            plt.legend(loc='lower right')
            plt.title("Feature Importance Plot")
            plt.show()
            st.pyplot(fig)



    # =======================================================================================================================================================
    # ========================================================= Why PCA =====================================================================================
    # =======================================================================================================================================================
    bal,bi,imbal = st.columns((1.4,0.1,1))

    with bal:
        with st.container():
            pass
            st.markdown(''' ## What is a Pipeline ?

**Pipelines** sequentially apply a list of transforms and a final estimator.Intermediate steps of the pipeline must be transforms, that is, they must implement `fit` and `transform` methods. The final estimator only needs to implement `fit`.
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a `'__'`, as in the example below. A step's estimator may be replaced entirely by setting the parameter with its name
to another estimator, or a transformer removed by setting it to `'passthrough'` or `None`.
Most people hack together models without pipelines, but pipelines have some important benefits. Those include:
- **Cleaner Code:** Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step. This will help in avoiding data leakage.
- **Fewer Bugs:** There are fewer opportunities to misapply a step or forget a preprocessing step.
- **Easier to Productionize:** It can be surprisingly hard to transition a model from a prototype to something deployable at scale.
- **More Options for Model Validation:** All parameters can be passed at once to the pipeline steps like PCA , Model Building , Scaling etc. .

Pipelines are valuable for cleaning up machine learning code and avoiding errors, and are especially useful for workflows with sophisticated data preprocessing.
            ''')

    with imbal:
        with st.container():
            st.write("")
            st.write("")

            imb = st.radio("",['Balanced Pipeline','Imbalance Pipeline'],0)

            if imb == 'Imbalance Pipeline':
                t = '<p style="color:red; font-size: 28px;">Pipeline For Imbalanced Classes</p>'
                st.markdown(t, unsafe_allow_html=True)
                st.markdown('''

        <div class="output_subarea output_html rendered_html output_result" dir="auto"><style>#sk-965986a2-f7e6-499a-b39e-ff3be569b002 {color: black;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 pre{padding: 0;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable {background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-estimator:hover {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-item {z-index: 1;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-parallel-item:only-child::after {width: 0;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-965986a2-f7e6-499a-b39e-ff3be569b002 div.sk-container {display: inline-block;position: relative;}</style><div id="sk-965986a2-f7e6-499a-b39e-ff3be569b002" class"sk-top-container"=""><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d9aa7a3a-615c-49c2-a4e1-64615f94702a" type="checkbox"><label class="sk-toggleable__label" for="d9aa7a3a-615c-49c2-a4e1-64615f94702a">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntransformer',
                         ColumnTransformer(remainder='passthrough',
                                           transformers=[('Numeric',
                                                          KNNImputer(),
                                                          Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')),
                                                         ('Categorical',
                                                          Pipeline(steps=[('cat_imputer',
                                                                           SimpleImputer(strategy='most_frequent')),
                                                                          ('cat_encoding',
                                                                           OneHotEncoder(drop='first'))]),
                                                          Index([], dtype='object'))])),
                        ('minmaxscaler', MinMaxScaler()), ('pca', PCA(n_components=12)),
                        ('smote', SMOTE(random_state=1, sampling_strategy='minority')),
                        ('Model', Model())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="dfb7081c-f09b-42e1-b47e-bb554b2e37c9" type="checkbox"><label class="sk-toggleable__label" for="dfb7081c-f09b-42e1-b47e-bb554b2e37c9">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder='passthrough',
                          transformers=[('Numeric', KNNImputer(),
                                         Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')),
                                        ('Categorical',
                                         Pipeline(steps=[('cat_imputer',
                                                          SimpleImputer(strategy='most_frequent')),
                                                         ('cat_encoding',
                                                          OneHotEncoder(drop='first'))]),
                                         Index([], dtype='object'))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="18917d44-ba2a-40e9-9129-373096147cf7" type="checkbox"><label class="sk-toggleable__label" for="18917d44-ba2a-40e9-9129-373096147cf7">Numeric</label><div class="sk-toggleable__content"><pre>Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality'],
              dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9d30486d-3aec-4eb3-8e5c-259f262596aa" type="checkbox"><label class="sk-toggleable__label" for="9d30486d-3aec-4eb3-8e5c-259f262596aa">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="665c7bc8-ae80-4a1f-a37b-3937b3d60589" type="checkbox"><label class="sk-toggleable__label" for="665c7bc8-ae80-4a1f-a37b-3937b3d60589">Categorical</label><div class="sk-toggleable__content"><pre>Index([], dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c95c2466-e267-4b13-bfd9-e4a2ba1b33dd" type="checkbox"><label class="sk-toggleable__label" for="c95c2466-e267-4b13-bfd9-e4a2ba1b33dd">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="9ffb8bde-d335-40d5-ae5d-a12c9f02b2ce" type="checkbox"><label class="sk-toggleable__label" for="9ffb8bde-d335-40d5-ae5d-a12c9f02b2ce">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop='first')</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ff39a13f-c1ae-4c28-8e90-246b16537e29" type="checkbox"><label class="sk-toggleable__label" for="ff39a13f-c1ae-4c28-8e90-246b16537e29">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2051673c-72eb-4b18-8985-82361a467d00" type="checkbox"><label class="sk-toggleable__label" for="2051673c-72eb-4b18-8985-82361a467d00">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0168020f-6535-42fb-a9b0-9c9429487023" type="checkbox"><label class="sk-toggleable__label" for="0168020f-6535-42fb-a9b0-9c9429487023">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1b5dfe2c-e4b1-413c-9252-e9f2699f2249" type="checkbox"><label class="sk-toggleable__label" for="1b5dfe2c-e4b1-413c-9252-e9f2699f2249">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=12)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a9116615-95de-4e33-a12a-0c428be63466" type="checkbox"><label class="sk-toggleable__label" for="a9116615-95de-4e33-a12a-0c428be63466">SMOTE</label><div class="sk-toggleable__content"><pre>SMOTE(random_state=1, sampling_strategy='minority')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="aa6e972a-f359-467e-b448-98ecf6528066" type="checkbox"><label class="sk-toggleable__label" for="aa6e972a-f359-467e-b448-98ecf6528066">Model</label><div class="sk-toggleable__content"><pre>Model()</pre></div></div></div></div></div></div></div></div>

            ''',unsafe_allow_html=True)

            else:
                t = '<p style="color:red;font-size: 28px;">Pipeline For Balanced Classes</p>'
                st.markdown(t, unsafe_allow_html=True)
                st.markdown('''
            <div class="output_subarea output_html rendered_html output_result" dir="auto"><style>#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e {color: black;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e pre{padding: 0;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable {background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-estimator:hover {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-item {z-index: 1;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-parallel-item:only-child::after {width: 0;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e div.sk-container {display: inline-block;position: relative;}</style><div id="sk-09fc030d-1fb4-471f-bb5a-e7ad3d85b77e" class"sk-top-container"=""><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3fc52e7f-cef0-4022-8167-d9180200b176" type="checkbox"><label class="sk-toggleable__label" for="3fc52e7f-cef0-4022-8167-d9180200b176">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('columntranformation',
                             ColumnTransformer(remainder='passthrough',
                                               transformers=[('Numeric',
                                                              KNNImputer(),
                                                              Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')),
                                                             ('Categorical',
                                                              Pipeline(steps=[('cat_imputer',
                                                                               SimpleImputer(strategy='most_frequent')),
                                                                              ('cat_encoding',
                                                                               OneHotEncoder(drop='first'))]),
                                                              Index([], dtype='object'))])),
                            ('scaler', MinMaxScaler()), ('pca', PCA(n_components=12)),
                            ('model',
                             Model())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a52cc582-3160-4269-891b-ab7d46d7181c" type="checkbox"><label class="sk-toggleable__label" for="a52cc582-3160-4269-891b-ab7d46d7181c">columntranformation: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder='passthrough',
                              transformers=[('Numeric', KNNImputer(),
                                             Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')),
                                            ('Categorical',
                                             Pipeline(steps=[('cat_imputer',
                                                              SimpleImputer(strategy='most_frequent')),
                                                             ('cat_encoding',
                                                              OneHotEncoder(drop='first'))]),
                                             Index([], dtype='object'))])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ecfe53b2-31ee-4c5f-9321-35e553ad7e70" type="checkbox"><label class="sk-toggleable__label" for="ecfe53b2-31ee-4c5f-9321-35e553ad7e70">Numeric</label><div class="sk-toggleable__content"><pre>Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                   'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                   'pH', 'sulphates', 'alcohol', 'quality'],
                  dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0f37bc6c-3f7b-4142-9a6d-41392201012c" type="checkbox"><label class="sk-toggleable__label" for="0f37bc6c-3f7b-4142-9a6d-41392201012c">KNNImputer</label><div class="sk-toggleable__content"><pre>KNNImputer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="aada903c-d6d6-4a2e-bfb7-8e27879189e7" type="checkbox"><label class="sk-toggleable__label" for="aada903c-d6d6-4a2e-bfb7-8e27879189e7">Categorical</label><div class="sk-toggleable__content"><pre>Index([], dtype='object')</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5e6d1826-feb8-4fe9-a7d1-e25ef25d769e" type="checkbox"><label class="sk-toggleable__label" for="5e6d1826-feb8-4fe9-a7d1-e25ef25d769e">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy='most_frequent')</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1f87e7dd-bd06-49c0-b47a-7ecede5f8cc1" type="checkbox"><label class="sk-toggleable__label" for="1f87e7dd-bd06-49c0-b47a-7ecede5f8cc1">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop='first')</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="90e72020-58ae-44c4-87fe-cc058ecbad63" type="checkbox"><label class="sk-toggleable__label" for="90e72020-58ae-44c4-87fe-cc058ecbad63">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="e6577c91-8c2f-418b-b805-41a59e80d959" type="checkbox"><label class="sk-toggleable__label" for="e6577c91-8c2f-418b-b805-41a59e80d959">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3d6e01d2-0948-4807-8c80-c916aceef3dd" type="checkbox"><label class="sk-toggleable__label" for="3d6e01d2-0948-4807-8c80-c916aceef3dd">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="a4c4ddd8-eb12-42e1-a17a-26a3b4f1ff9b" type="checkbox"><label class="sk-toggleable__label" for="a4c4ddd8-eb12-42e1-a17a-26a3b4f1ff9b">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=12)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d69272fd-f66a-4236-a46f-15801bb9c854" type="checkbox"><label class="sk-toggleable__label" for="d69272fd-f66a-4236-a46f-15801bb9c854">model</label><div class="sk-toggleable__content"><pre>Model()</pre></div></div></div></div></div></div></div></div>
                ''', unsafe_allow_html=True)

    st.write("---")

    c0,c1,c2,c3 = st.columns((0.02,0.60,0.02,1))
    o0,o1,o2,o3 = st.columns((0.9,0.55,0.55,0.15))

    with o1:
        scale = st.checkbox('PCA without Scaling',key = 's')
        exp1 = st.expander("Why Scaling ? ")
        with exp1:

            st.caption('''- Many machine learning algorithms perform better or converge faster when features are on a relatively similar scale and/or close to normally distributed.

- Machine learning algorithm just sees number , if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort.''')

    with o2:
        smote = st.checkbox('Oversample Minority Class',key = 'smote')
        exp2 = st.expander("Why OverSample Minority class ? ")

        with exp2:

            st.caption('''- Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce errors.
- However, if the data set in imbalance then In such cases, you get a **pretty high accuracy just by predicting the majority class, but you fail to capture the minority class**, which is most often the point of creating the model in the first place.
- **SMOTE** is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.''')
    with o3:

        dimension = st.select_slider("Dimension:",['3D','2D'],"2D",key = 1)

        if not scale:
            pca_pipeline = Pipeline(steps = [("ct",column_tranformation),('sclaer',scaler),('pca',pca)])
            pc_df = pd.DataFrame(pca_pipeline.fit_transform(X_train),columns = ['PC'+str(i) for i in range(1,13)],index = X_train.index)
            pc_df = pc_df.iloc[:,0:3]
            labels = pd.DataFrame(ord_encoder.inverse_transform(y_train),index = y_train.index,columns = ['discrete_quality'])
            full_df = pd.concat([pc_df,labels],axis = 1)

            if smote:
                pca_pipeline = make_pipeline(column_tranformation,scaler,pca,over_samp)
                ox,oy = pca_pipeline.fit_resample(X_train,y_train)
                pc_df = pd.DataFrame(ox,columns = ['PC'+str(i) for i in range(1,13)])
                pc_df = pc_df.iloc[:,0:3]
                labels = pd.DataFrame(ord_encoder.inverse_transform(oy),columns = ['discrete_quality'])
                full_df = pd.concat([pc_df,labels],axis = 1)

        else:
            pca_pipeline = Pipeline(steps = [("ct",column_tranformation),('pca',pca)])
            pc_df = pd.DataFrame(pca_pipeline.fit_transform(X_train),columns = ['PC'+str(i) for i in range(1,13)],index = X_train.index)
            pc_df = pc_df.iloc[:,0:3]
            labels = pd.DataFrame(ord_encoder.inverse_transform(y_train),index = y_train.index,columns = ['discrete_quality'])
            full_df = pd.concat([pc_df,labels],axis = 1)

            if smote:
                pca_pipeline = make_pipeline(column_tranformation,pca,over_samp)
                ox,oy = pca_pipeline.fit_resample(X_train,y_train)
                pc_df = pd.DataFrame(ox,columns = ['PC'+str(i) for i in range(1,13)])
                pc_df = pc_df.iloc[:,0:3]
                labels = pd.DataFrame(ord_encoder.inverse_transform(oy),columns = ['discrete_quality'])
                full_df = pd.concat([pc_df,labels],axis = 1)


    with c1:
        st.header(" WHY PCA ? ")

        st.markdown('''Principal component analysis (PCA) is one of a family of techniques for taking high-dimensional data, and using the dependencies between the variables to represent it in a more tractable, lower-dimensional basis, without losing too much information.
**Principal Component Analysis (PCA)** is Statistical procedure to convert observations of possibly correlated variables into 'Principal Components' that are:
- Independent of each other (Uncorrelated)
- That Captures maximum information in first few components

- **UseCases** :
    - Dimensionality reduction
    - Data visualization and Exploratory Data Analysis
    - Create uncorrelated features/variables that can be an input to a prediction model
    - Uncovering latent variables/themes/concepts
    - Noise reduction in the dataset
''')
    with o0:
        fig = ex.line(y = pca_pipeline.named_steps.pca.explained_variance_ratio_,x = ['PC'+str(i) for i in range(1,13,1)],labels = {'x':'Components','y':'Explained Variance (%)'})
        fig.update_layout(showlegend=False,width = 600,height = 400,title = "Scree Plot",title_x = 0.5)
        st.plotly_chart(fig)

        with st.expander("Scree Plot"):
            st.caption('''**Screen Plot** helps in identifying the components required for building a model.After a certain point the variance tends to decreese very minimal which indicates the number of components.This graph is generally defined as Elbow Plot or Scree Plot''')


    # fig = ex.line(pca_pipeline.named_steps.pca.explained_variance_ratio_,labels = {'index':'Components','value':'Cumulative Explained Variance'})
    # fig.update_layout(showlegend=False)
    # st.plotly_chart(fig)


    with c3:

        st.write(" ")
        if dimension == '3D':

            fig = ex.scatter_3d(full_df, x='PC1', y='PC2', z='PC3',
                      color='discrete_quality',opacity=0.7)

            fig.update_layout(width = 600,height = 600,title = "Visualizing the Quality of wine - 3D",title_x = 0.5,title_y = 0.95)

            st.plotly_chart(fig,use_container_width=True)

        elif dimension == '2D':

            fig = ex.scatter(full_df, x='PC1', y='PC2',
                      color='discrete_quality',opacity=0.7)

            fig.update_layout(width = 600,height = 600,title = "Visualizing the Quality of wine - 2D",title_x = 0.5,title_y = 0.95)

            st.plotly_chart(fig,use_container_width=True)
    st.write("---")

    # st.markdown("6308, 6075, 1236 , 1328 ")

    st.markdown(" ## Models Evaluation")

    st.write('Test Data')

    ind = np.random.randint(10,len(X_test)-10)
    input = X_test.iloc[ind:ind+1,:]
    # st.write(input)
    #
    # st.write(input.index[0])

    dis = inputdata1()[inputdata1().index == input.index[0]]
    st.dataframe(dis[X_train.columns])

    # st.selectbox("Choose the model",['Logistic Regression','Decision Tree Classifier',
    # 'Random Forrest Classifier','XGBoost Classifier','Neural Network','Adaboost Classifier','Gradient Boost Classifier'])

    pred = st.button("Predict Quality")

    # loaded_model = joblib.load('decisiontreeclassifierQ-Imbalance.pkl')
    # model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
    # si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1

    
    m1,m2,m3,m4,m5,m6,m7 = st.columns((1,1,1,1,1,1,1))

    if pred:
        with m1:
            st.metric('Actual Value',np.array(dis.discrete_quality)[0],)
        with m2:
            loaded_model = joblib.load('logisticregressionQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('Logistic Regression',model_output,si)
        with m3:
            loaded_model = joblib.load('decisiontreeclassifierQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('Decision Tree Classifier',model_output,si)
        with m4:
            loaded_model = joblib.load('randomforestclassifierQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('Random Forest Classifier',model_output,si)
        with m5:
            loaded_model = joblib.load('xgbclassifierQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('XGBoost Classifier',model_output,si)
        with m6:
            loaded_model = joblib.load('adaboostclassifierQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('AdaBoost Classifier',model_output,si)
        with m7:
            loaded_model = joblib.load('gradientboostingclassifierQ-Imbalance.pkl')
            model_output = "High" if loaded_model.predict(input)==0 else ("Low" if loaded_model.predict(input)==1 else 'Medium')
            si = 1 if model_output ==  np.array(dis.discrete_quality)[0] else -1
            st.metric('Gradient Boost Classifier',model_output,si)

    else:
        st.metric('Actual Value',np.array(dis.discrete_quality)[0],)
