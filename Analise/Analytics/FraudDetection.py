# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore")
df = pd.read_csv('creditcard.csv')


#Lendo o csv
print(df.head())

print(df.describe())

#Verificando se existe valores nulos
print(df.isnull().sum().max())

#ver colunas
print(df.columns)

# As classes são muito desequilibradas, precisamos resolver esse problema mais tarde.
print(round(df['Class'].value_counts()[0]/len(df) * 100,2), '% Sem fraudes')
print(round(df['Class'].value_counts()[1]/len(df) * 100,2), '% Com fraude')


colors = ["#007bff", "#dc3545"]

sns.countplot(x='Class', data=df, palette=colors)
plt.title('Distribuições de classe \n (0: Sem Fraude || 1: Fraude)', fontsize=14)
plt.show()


#Distribuição do valor da transação
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribuição do valor da transação', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')

plt.show()


# Como nossas classes são altamente distorcidas, devemos torná-las equivalentes para ter uma distribuição normal das classes.
# Vamos embaralhar os dados antes de criar as subamostras

df = df.sample(frac=1)

# Quantidade de classes de fraude 492 linhas.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Misturar linhas de dataframe
new_df = normal_distributed_df.sample(frac=1, random_state=42)

print(new_df.head())
print('Distribuição das classes no conjunto de dados da subamostra')
print(new_df['Class'].value_counts()/len(new_df))


sns.countplot(x='Class', data=new_df, palette=colors)
plt.title('Classes igualmente distribuídas', fontsize=14)
plt.show()

"""
A correlação é uma estatística muito comum e muito utilizada.
Desconsiderando casos de falsas correlações, como o exemplo de Gustav Fischer que provou, na década de 30,
a existência de uma correlação positiva entre o tamanho da população da cidade de Oldenburg e o número de cegonhas.
Ele fez isso não porque acreditasse no mito infantil, mas exatamente para alertar para um erro muito comum quando o assunto é economia:
confundir correlação com causa. Fischer provou que a população e o número de cegonhas aumentaram ao longo do período de estudo.
O resultado não significa que o crescente número de cegonhas causou o aumento observado na população, obviamente.
"""

# Certifique-se de usar a subamostra em nossa correlação

# DataFrame inteiro
# Matriz de correlação
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

# Matriz de correlação dataframe completo
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
ax1.set_title("Matriz de Correlação Desequilibrada \n (não use para referência)", fontsize=14)

# Matriz de correlação dataframe criado
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax2)
ax2.set_title('Matriz de correlação de subamostra \n (usar para referência)', fontsize=14)
plt.show()

# Calculo das correlacoes dos retornos e criação dos gráficos.
rets = df.corr()
corr = rets.corr()
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);
plt.show()

# Calculo das correlacoes dos retornos e criação dos gráficos.
rets = new_df.corr()
corr = rets.corr()
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);
plt.show()

"""
correlacoes
"""


# Correlações negativas com nossa classe (quanto menor o valor do nosso recurso, mais provável será uma transação fraudulenta)
f, axes = plt.subplots(ncols=14, figsize=(20, 4))
for x in range(14):
    col = "V{}".format(x+1)
    sns.boxplot(x="Class", y=col, data=new_df, palette=colors, ax=axes[x])
    axes[x].set_title(col)
plt.show()

f, axes = plt.subplots(ncols=14, figsize=(20, 4))
for x in range(14):
    col = "V{}".format(x+15)
    sns.boxplot(x="Class", y=col, data=new_df, palette=colors, ax=axes[x])
    axes[x].set_title(col)
plt.show()


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Correlações negativas com nossa classe (quanto menor o valor do nosso recurso, mais provável será uma transação fraudulenta)
sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Classe de correlação negativa')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Classe de correlação negativa')


sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Classe de correlação negativa')


sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Classe de correlação negativa')

plt.show()


f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Correlações positivas (quanto maior o recurso, aumenta a probabilidade de que seja uma transação fraudulenta)
sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V11 vs Classe de correlação positiva')

sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Classe de correlação positiva')


sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V2 vs Classe de correlação positiva')


sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V19 vs Classe de correlação positiva')

plt.show()


from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)


v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()

f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

colors = ["#007bff", "#dc3545"]

# Boxplots com outliers removidos

# Recurso V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Recurso V12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

# Recurso V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'),
            fontsize=14)

plt.show()

