import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import math

def summarize_columns(df, target_col):
    """
    สรุปประเภทของคอลัมน์ใน DataFrame
    - แยกเป็น numeric, categorical และ target

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการวิเคราะห์
    target_col (str): ชื่อคอลัมน์ที่เป็น Target

    Returns:
    dict: {'numeric': [num_cols], 'categorical': [cat_cols], 'target': target_col}
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in num_cols:
        num_cols.remove(target_col)
    elif target_col in cat_cols:
        cat_cols.remove(target_col)

    return {"numeric": num_cols, "categorical": cat_cols, "target": target_col}

def handle_missing_values(df, plot_missing=False):
    """
    จัดการกับ Missing Values และสามารถ plot ปริมาณ Missing Value (%)
    - ถ้า column มี missing มากกว่า 20% ให้ดรอป column นั้น
    - ถ้าเป็น column เลข (numeric) และมี missing น้อยกว่า 20% ให้ fill ด้วย median
    - ถ้าเป็น column categorical ให้ fill ด้วย mode
    
    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการจัดการกับ missing values
    plot_missing (bool): หากเป็น True จะแสดงกราฟ missing value
    
    Returns:
    pd.DataFrame: DataFrame ที่ถูกจัดการกับ missing values
    """
    # ตรวจสอบอัตราการ missing ในแต่ละ column
    missing_percentage = df.isnull().mean() * 100
    
    if plot_missing:
        plt.figure(figsize=(10, 5))
        missing_percentage[missing_percentage > 0].sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel('Missing Value (%)')
        plt.ylabel('Columns')
        plt.title('Missing Value Percentage per Column')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()
    
    # ลบคอลัมน์ที่มี missing มากกว่า 20%
    columns_to_drop = missing_percentage[missing_percentage > 20].index
    df = df.drop(columns=columns_to_drop)

    # แบ่งเป็น numeric กับ categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values สำหรับ numeric columns ด้วย median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing values สำหรับ categorical columns ด้วย mode
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

def cap_outliers(df, ignore_feature=None, plot=False):
    """
    Cap outliers in any numeric column using the 1.5 * IQR rule, excluding the target_feature.
    Optionally plot boxplots before and after capping for affected columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        ignore_feature (str, optional): Column to be ignored when capping outliers.
        plot (bool): Whether to plot boxplots before and after capping.
    
    Returns:
        pd.DataFrame: DataFrame with outliers capped.
    """
    df = df.copy()
    affected_columns = []
    original_data = df.copy()
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col == ignore_feature:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0] > 0:
            affected_columns.append(col)
            
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    if plot and affected_columns:
        num_cols = len(affected_columns)
        fig, axes = plt.subplots(num_cols, 2, figsize=(10, 5 * num_cols))
        
        if num_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, affected_columns):
            ax[0].boxplot(original_data[col], vert=False)
            ax[0].set_title(f"{col} Before Capping")
            
            ax[1].boxplot(df[col], vert=False)
            ax[1].set_title(f"{col} After Capping")
        
        plt.tight_layout()
        plt.show()
    
    return df

def explore_data(df):
    """
    Explore the dataset by visualizing missing values, numerical distributions, and categorical distributions.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    """
    def plot_missing_values(df):
        missing_percentage = df.isnull().mean() * 100
        missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
        
        if not missing_percentage.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=missing_percentage.index, y=missing_percentage.values, color='navy')
            plt.xticks(rotation=90)
            plt.ylabel("% Missing Values")
            plt.title("Missing Values Percentage by Column")
            plt.show()
        else:
            print("No missing values in the dataset.")
    
    def plot_numeric_features(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(x=df[col], ax=axes[0])
            axes[0].set_title(f"Boxplot of {col}")
            
            sns.histplot(df[col], kde=True, ax=axes[1])
            axes[1].set_title(f"Distribution of {col}")
            
            plt.tight_layout()
            plt.show()
    
    def plot_categorical_features(df):
        categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
        num_cats = len(categorical_cols)
        
        if num_cats > 0:
            rows = (num_cats // 4) + (num_cats % 4 > 0)
            fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
            axes = axes.flatten()
            
            for i, col in enumerate(categorical_cols):
                sns.countplot(x=df[col], order=df[col].value_counts().index, ax=axes[i], color='navy')
                axes[i].set_title(f"Countplot of {col}")
                axes[i].tick_params(axis='x', rotation=90)
            
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            plt.show()
        else:
            print("No categorical features in the dataset.")
    
    print("Exploring Missing Values:")
    plot_missing_values(df)
    print("\nExploring Numeric Features:")
    plot_numeric_features(df)
    print("\nExploring Categorical Features:")
    plot_categorical_features(df)


def plot_feature_relationships(df, target_col):
    """
    วาดกราฟเปรียบเทียบระหว่าง features และ target
    - Numeric: Distribution Plot
    - Categorical: Bar Chart

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการวิเคราะห์
    target_col (str): ชื่อคอลัมน์ที่เป็น Target
    """
    summary = summarize_columns(df, target_col)
    
    num_cols = summary['numeric']
    cat_cols = summary['categorical']

    # สร้าง subplot สำหรับ Numeric Features
    if num_cols:
        n = len(num_cols)
        ncols = 3  # กำหนดให้มี 3 คอลัมน์
        nrows = math.ceil(n / ncols)  # คำนวณจำนวนแถวที่ต้องการ
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4))  # ขนาดของภาพ
        axes = axes.flatten()  # ทำให้ axes เป็น list แทนที่จะเป็น array 2D
        for i, col in enumerate(num_cols):
            sns.kdeplot(df[df[target_col] == 0][col], label="Non-Default", shade=True, color="blue", ax=axes[i])
            sns.kdeplot(df[df[target_col] == 1][col], label="Default", shade=True, color="red", ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].legend()
        # ปิด subplot ที่เหลือถ้าไม่พอ
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()  # ทำให้ spacing ของ subplot ไม่ทับกัน
        plt.show()

    # Plot Bar Chart สำหรับ Categorical Features
    hue_colors = {1: '#CB3335', 0: '#477CA8'}
    if cat_cols:
        n = len(cat_cols)
        ncols = 6  # กำหนดให้มี 6 คอลัมน์
        nrows = math.ceil(n / ncols)  # คำนวณจำนวนแถวที่ต้องการ
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4))  # ขนาดของภาพ
        axes = axes.flatten()  # ทำให้ axes เป็น list แทนที่จะเป็น array 2D
        for i, col in enumerate(cat_cols):
            sns.countplot(x=col, hue=target_col, data=df, palette=hue_colors, ax=axes[i])
            axes[i].set_title(f"Count Plot of {col} by {target_col}")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        # ปิด subplot ที่เหลือถ้าไม่พอ
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()  # ทำให้ spacing ของ subplot ไม่ทับกัน
        plt.show()

def plot_correlation_heatmap(df, target_col):
    """
    แสดง Heatmap ของ correlation ระหว่างตัวแปร numeric กับ Target

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการวิเคราะห์
    target_col (str): ชื่อคอลัมน์ที่เป็น Target
    """
    summary = summarize_columns(df, target_col)
    num_cols = summary['numeric'] + [target_col]

    corr_matrix = df[num_cols].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def chi_square_test(df, target_col):
    """
    คำนวณ Chi-Square Test เพื่อดูว่า categorical variable ใดมีความสัมพันธ์กับ target

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการวิเคราะห์
    target_col (str): ชื่อคอลัมน์ที่เป็น Target

    Returns:
    pd.DataFrame: ตารางแสดง p-value ของแต่ละ categorical variable
    """
    summary = summarize_columns(df, target_col)
    cat_cols = summary['categorical']

    chi2_results = {}
    alpha = 0.05  # ระดับนัยสำคัญ

    for col in cat_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        chi2_results[col] = p

    chi2_df = pd.DataFrame.from_dict(chi2_results, orient='index', columns=['p-value']).sort_values(by='p-value')
    
    print("\n🔍 Significant Variables:")
    print(chi2_df[chi2_df['p-value'] < alpha])

    return chi2_df


def encode_categorical_features(df, target_col):
    """
    ทำ One-Hot Encoding กับ categorical variables และคืนค่าเป็น DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame ที่ต้องการ encoding
    target_col (str): ชื่อคอลัมน์ที่เป็น Target

    Returns:
    pd.DataFrame: DataFrame ที่ถูก encode แล้ว
    """
    summary = summarize_columns(df, target_col)
    cat_cols = summary['categorical']
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # ✅ ใช้ sparse_output=False แทน sparse=False
    encoded_cat = encoder.fit_transform(df[cat_cols])

    # สร้าง DataFrame ของ One-Hot Encoded Features
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    # รวมกับ numeric columns และ target
    final_df = pd.concat([df[summary['numeric']], encoded_df, df[target_col]], axis=1)

    return final_df



def create_model_pipeline(X_train, y_train, models_dict, cv_folds=5):
    """
    ค้นหาโมเดลที่ดีที่สุดโดยใช้ ROC-AUC Score และทำ Grid Search หา hyperparameters ที่ดีที่สุด

    Parameters:
    X_train (pd.DataFrame): ข้อมูลฟีเจอร์ของชุด training
    y_train (pd.Series): ข้อมูล target ของชุด training
    models_dict (dict): พจนานุกรมที่เก็บโมเดลและพารามิเตอร์ของแต่ละโมเดล
    cv_folds (int): จำนวน folds สำหรับ Cross-Validation (ค่า default = 5)

    Returns:
    tuple: ชื่อโมเดลที่ดีที่สุด, ค่า ROC-AUC Score ที่ดีที่สุด, pipeline ที่ถูกฝึกแล้ว
    """
    best_model = None
    best_score = 0
    results = {}

    # ใช้ Stratified K-Fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ใช้ ROC-AUC เป็น metric
    roc_auc_scorer = make_scorer(roc_auc_score)

    # 1️⃣ ค้นหาโมเดลที่ดีที่สุด (Baseline Model)
    for name, model_info in models_dict.items():
        print(f'🔍 Evaluating Model: {name} ...')

        model = model_info['model']

        # ใช้ SMOTE เพื่อแก้ปัญหาข้อมูล imbalance
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # สร้าง Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # คำนวณค่า ROC-AUC Score
        roc_auc_scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=cv, scoring=roc_auc_scorer)
        mean_roc_auc = roc_auc_scores.mean()
        results[name] = mean_roc_auc

        print(f'ROC-AUC Score: {mean_roc_auc:.4f}')

        # อัปเดตโมเดลที่ดีที่สุด
        if mean_roc_auc > best_score:
            best_score = mean_roc_auc
            best_model = name

    print(f"\n🏆 Best Model: {best_model} with ROC-AUC Score: {best_score:.4f}")

    # 2️⃣ ทำ Grid Search กับโมเดลที่ดีที่สุด
    best_model_info = models_dict[best_model]
    best_model_instance = best_model_info['model']
    param_grid = {f'classifier__{key}': value for key, value in best_model_info['params'].items()}

    print(f"\n🔍 Performing Grid Search for {best_model} ...")

    # สร้าง Pipeline สำหรับ Grid Search
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', best_model_instance)
    ])

    # ใช้ GridSearchCV เพื่อหาค่า hyperparameter ที่ดีที่สุด
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=roc_auc_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # ได้ค่าพารามิเตอร์ที่ดีที่สุด
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_final_score = grid_search.best_score_

    print(f"✅ Best Parameters for {best_model}: {best_params}")
    print(f"🏆 Final ROC-AUC Score: {best_final_score:.4f}")

    return best_model, best_final_score, best_pipeline


# def create_model_pipeline(X_train, y_train, model=RandomForestClassifier(random_state=42), cv_folds=5, param_grid=None):
#     """
#     สร้าง Pipeline สำหรับการทำ Standardization, SMOTE, Feature Selection (SelectFromModel),
#     การทำ GridSearch เพื่อหาพารามิเตอร์ที่ดีที่สุด และ Cross-Validation เพื่อคำนวณ F2 Score

#     Parameters:
#     X_train (pd.DataFrame): DataFrame ที่เก็บฟีเจอร์ในชุด training
#     y_train (pd.Series): Series ที่เก็บ target ในชุด training
#     model (sklearn model object): โมเดลที่ต้องการใช้ (ค่า default = RandomForestClassifier)
#     cv_folds (int): จำนวน folds สำหรับ Cross-Validation (ค่า default = 5)
#     param_grid (dict): พจนานุกรมของพารามิเตอร์ที่ต้องการค้นหาจาก Grid Search (ค่า default = None)

#     Returns:
#     tuple: ค่าเฉลี่ยของ F2 Score จากการทำ Cross-Validation และ pipeline ที่ถูกฝึกแล้ว
#     """
#     # สร้าง Pipeline
#     smote = SMOTE(sampling_strategy='auto', random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)

#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Standardize data
#         ('classifier', model)          # ใส่โมเดลที่เลือก
#     ])
    
#     # หากไม่ได้ระบุ param_grid ให้ใช้ค่าพารามิเตอร์เริ่มต้น
#     if param_grid is None:
#         param_grid = {
#             'classifier__n_estimators': [100, 200],
#             'classifier__max_depth': [None, 10, 20]
#         }
    
#     # สร้าง F2 Score Scorer
#     f2_scorer = make_scorer(fbeta_score, beta=2)
    
#     # ใช้ StratifiedKFold เพื่อให้ข้อมูลแต่ละ fold มีสัดส่วนของ target ที่เท่ากัน
#     cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

#     # สร้าง GridSearchCV
#     grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f2_scorer, n_jobs=-1)
    
#     # ทำการ Grid Search เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด
#     grid_search.fit(X_train, y_train)

#     # ค่าเฉลี่ยของ F2 Score จาก GridSearchCV
#     mean_f2_score = grid_search.best_score_
    
#     # แสดงผลลัพธ์
#     print(f"🔍 F2 Score (จากการทำ {cv_folds}-fold cross-validation): {mean_f2_score:.4f}")
#     print(f"🔍 พารามิเตอร์ที่ดีที่สุด: {grid_search.best_params_}")
    
#     # ฝึกโมเดลด้วยพารามิเตอร์ที่ดีที่สุด
#     best_pipeline = grid_search.best_estimator_

#     return mean_f2_score, best_pipeline


def get_model_accuracy(model, params, X, y, cv_folds=5):
    # กำหนด StratifiedKFold สำหรับ cross-validation
    stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # F2 Score
    f2_scorer = make_scorer(fbeta_score, beta=2)  # ใช้ F2-score

    # สร้าง GridSearchCV ด้วย StratifiedKFold
    grid = GridSearchCV(model,
                        params,
                        scoring=f2_scorer,
                        cv=stratified_kfold,
                        error_score=0.)
    grid.fit(X, y)
    
    # เก็บผลลัพธ์
    best_score = grid.best_score_
    best_params = grid.best_params_
    fit_time = round(grid.cv_results_['mean_fit_time'].mean(), 3)
    score_time = round(grid.cv_results_['mean_score_time'].mean(), 3)
    

    print('Best F2 Score: {:.4f}'.format(grid.best_score_))
    print('Best Parameters: {}'.format(grid.best_params_))
    print('Average Time of Fit (s): {:.3f}'.format(grid.cv_results_['mean_fit_time'].mean()))
    print('Average Time to Score (s): {:.3f}'.format(grid.cv_results_['mean_score_time'].mean()))
    
    # Return ผลลัพธ์รวมถึง grid object ด้วย
    return {
        'best_score': best_score,
        'best_params': best_params,
        'fit_time': fit_time,
        'score_time': score_time,
        'grid': grid
    }

def compare_models(models, params, X, y):
    results = []
    
    # วน loop สำหรับแต่ละ model และเก็บผลลัพธ์
    for model_name, model in models.items():
        print(f"Running GridSearchCV for {model_name}...")
        result = get_model_accuracy(model, params[model_name], X, y)
        result['model'] = model_name  # เพิ่มชื่อ model
        results.append(result)
    
    # แปลงผลลัพธ์เป็น DataFrame เพื่อการเปรียบเทียบที่ง่ายขึ้น
    results_df = pd.DataFrame(results)
    return results_df


def plot_boxplots_comparison(df1, df2, num_cols):
    """
    วาด Boxplot เปรียบเทียบระหว่าง DataFrame 2 ตัว
    สำหรับแต่ละคอลัมน์ใน num_cols
    
    Parameters:
    df1 (pd.DataFrame): DataFrame ตัวแรก
    df2 (pd.DataFrame): DataFrame ตัวที่สอง
    num_cols (list): รายชื่อคอลัมน์ที่เป็น Numeric ในทั้งสอง DataFrame
    """
    n = len(num_cols)
    ncols = 4  # กำหนดให้มี 3 คอลัมน์
    nrows = math.ceil(n / ncols)  # คำนวณจำนวนแถวที่ต้องการ
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4))  # ขนาดของภาพ
    axes = axes.flatten()  # ทำให้ axes เป็น list แทนที่จะเป็น array 2D
    
    for i, col in enumerate(num_cols):
        # สร้าง DataFrame ใหม่เพื่อเพิ่มคอลัมน์ที่ระบุว่าเป็นของ df1 หรือ df2
        df1_copy = df1[[col]].copy()
        df1_copy['DataFrame'] = 'Before'
        
        df2_copy = df2[[col]].copy()
        df2_copy['DataFrame'] = 'After'
        
        # รวมทั้งสอง DataFrame
        combined_df = pd.concat([df1_copy, df2_copy])
        
        # วาด Boxplot
        sns.boxplot(x='DataFrame', y=col, data=combined_df, ax=axes[i], palette="Set1")
        axes[i].set_title(f"Boxplot of {col}")
    
    # ปิด subplot ที่เหลือถ้าไม่พอ
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()  # ทำให้ spacing ของ subplot ไม่ทับกัน
    plt.show()


def plot_status_comparison(status_series,name):
    """
    สร้าง bar chart เปรียบเทียบจำนวนของ Status 1 และ 0 โดยใช้สีตามที่กำหนด
    และแสดงเปอร์เซ็นต์บนหัวของแท่งใน bar chart
    
    Parameters:
    status_series (pd.Series): Series ที่มีข้อมูลของ Status (1 หรือ 0)
    """
    # กำหนดสี
    hue_colors = {1: '#CB3335', 0: '#477CA8'}
    
    # สร้าง bar chart
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=status_series, palette=hue_colors)

    # เปลี่ยนค่าใน x-axis
    ax.set_xticklabels(['Non-Default', 'Default'])

    # คำนวณและแสดงเปอร์เซ็นต์บนหัวแท่ง
    total = len(status_series)
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.text(p.get_x() + p.get_width() / 2, height + 0.05, f'{percentage:.2f}%', 
                ha='center', va='bottom', fontsize=12)
    
    # เพิ่มรายละเอียดในกราฟ
    plt.title(f"Count Plot of Status of {name}")
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
