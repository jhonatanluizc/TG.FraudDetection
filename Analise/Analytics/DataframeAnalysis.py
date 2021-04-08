""" Imports Libraries """

""" Data Processing, CSV file """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

""" Describe File """
def FileDescribe(currentFile):

    """ Read .csv file and get describe """
    print(currentFile.head())
    print(currentFile.describe())

    """ Verify Null values exits """
    print(currentFile.isnull().sum().max())

    """ View column names """
    print(currentFile.columns)

    """ Get class disposition """
    print(round(currentFile['Class'].value_counts()[0]/len(currentFile) * 100, 2), '% Sem fraudes')
    print(round(currentFile['Class'].value_counts()[1]/len(currentFile) * 100, 2), '% Com fraude')

""" Get File Distribution """
def Distribution(currentFile):

    """ Define colors """
    colors = ["#007bff", "#dc3545"]

    """ Configure display """
    sns.countplot(x='Class', data=currentFile, palette=colors)
    plt.title('Distribuições de classe \n (0: Sem Fraude || 1: Fraude)', fontsize=14)

    """ Show display """
    plt.show()

""" Distribution of the transaction value """
def TransactionDistribution(currentFile):

    """ Configure display """
    fig, ax = plt.subplots(1, 2, figsize=(18,4))

    amount_val = currentFile['Amount'].values
    time_val = currentFile['Time'].values

    sns.distplot(amount_val, ax=ax[0], color='r')
    ax[0].set_title('Distribuição do valor da transação', fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax[1], color='b')

    """ Show display """
    plt.show()

""" Shuffle data """
def ShuffleData(currentFile):
    return currentFile.sample(frac=1)

""" Date equalize """
def EqualizeData(currentFile):

    """ Get fraud and noFraud """
    fraud = currentFile.loc[currentFile['Class'] == 1]
    noFraud = currentFile.loc[currentFile['Class'] == 0][:492]

    """ Concat fraud and noFraud """
    currentFile = pd.concat([fraud, noFraud])

    """ Shuffle data """
    currentFile.sample(frac=1, random_state=42)

    return currentFile

""" Negative correlations """
def NegativeCorrelations(currentFile):

    """ Define colors """
    colors = ["#007bff", "#dc3545"]

    """ Configure display """
    f, axes = plt.subplots(ncols=4, figsize=(20, 4))

    sns.boxplot(x="Class", y="V17", data=currentFile, palette=colors, ax=axes[0])
    axes[0].set_title('V17 vs Classe de correlação negativa')

    sns.boxplot(x="Class", y="V14", data=currentFile, palette=colors, ax=axes[1])
    axes[1].set_title('V14 vs Classe de correlação negativa')

    sns.boxplot(x="Class", y="V12", data=currentFile, palette=colors, ax=axes[2])
    axes[2].set_title('V12 vs Classe de correlação negativa')

    sns.boxplot(x="Class", y="V10", data=currentFile, palette=colors, ax=axes[3])
    axes[3].set_title('V10 vs Classe de correlação negativa')

    """ Show display """
    plt.show()

""" Positive correlations """
def PositiveCorrelations(currentFile):

    """ Define colors """
    colors = ["#007bff", "#dc3545"]

    """ Configure display """
    f, axes = plt.subplots(ncols=4, figsize=(20, 4))

    # Correlações positivas (quanto maior o recurso, aumenta a probabilidade de que seja uma transação fraudulenta)
    sns.boxplot(x="Class", y="V11", data=currentFile, palette=colors, ax=axes[0])
    axes[0].set_title('V11 vs Classe de correlação positiva')

    sns.boxplot(x="Class", y="V4", data=currentFile, palette=colors, ax=axes[1])
    axes[1].set_title('V4 vs Classe de correlação positiva')

    sns.boxplot(x="Class", y="V2", data=currentFile, palette=colors, ax=axes[2])
    axes[2].set_title('V2 vs Classe de correlação positiva')

    sns.boxplot(x="Class", y="V19", data=currentFile, palette=colors, ax=axes[3])
    axes[3].set_title('V19 vs Classe de correlação positiva')

    plt.show()

""" Current Directory File and open File """
directoryFile = 'creditcard.csv'
file = pd.read_csv(directoryFile)

""" Script 'analysis dataset' """
FileDescribe(file)
Distribution(file)
# TransactionDistribution(file)
file = ShuffleData(file)
file = EqualizeData(file)
Distribution(file)
NegativeCorrelations(file)
PositiveCorrelations(file)

""" Remove columns """
col = file.columns

noRemove = ['V17', 'V14', 'V12', 'V10', 'V11', 'V4', 'V2', 'V19', 'Time', 'Amount', 'Class']

print(file.columns)
for x in col:
    if not x in noRemove:
        file = file.drop(columns=[x])

print(file.columns)

""" Save data analysis result """
file.to_csv('analysis.csv', encoding='utf-8', index=False)
