from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

global filename
global df, X_train, X_test, y_train, y_test
global rf

main = tk.Tk()
main.title("Red Wine Quality Prediction Using Machine Learning Techniques") 
main.geometry("1600x900")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Red Wine Quality Prediction Using Machine Learning Techniques', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

font1 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=250, y=600)

def plot_quality_distribution():
    # Plotting the pie chart for the quality column
    quality_counts = df['quality'].value_counts()
    labels = quality_counts.index
    sizes = quality_counts.values
    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(10, 7))
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.3))

    # Customize the text properties
    for text in texts:
        text.set_fontsize(12)
        text.set_color('black')

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_weight('bold')

    # Draw a circle at the center to make it look like a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Distribution of Wine Quality', fontsize=16, weight='bold')
    plt.show()


qualityButton = tk.Button(main, text="Graph 1 (Quality Distribution)", command=plot_quality_distribution, bg="orange", width=25)
qualityButton.place(x=50, y=650)
qualityButton.config(font=font1)

def plot_histograms():
    global df
    if df is None:
        messagebox.showerror("Error", "Please upload a dataset first.")
        return

    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 4, i)
        df[column].hist(bins=20, color='skyblue')
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

graph2Button = tk.Button(main, text="Graph 2 (Histograms)", command=plot_histograms, bg="light salmon", width=20)
graph2Button.place(x=350, y=650)
graph2Button.config(font=font1)


def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["quality"], axis=1))
    y = np.array(df["quality"])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="light green", width=15)
splitButton.place(x=50, y=700)
splitButton.config(font=font1)


def NaiveBayes():
    global nb_acc, nb_precision, nb_recall, nb_f1
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, y_pred)
    nb_precision = precision_score(y_test, y_pred, average='weighted')
    nb_recall = recall_score(y_test, y_pred, average='weighted')
    nb_f1 = f1_score(y_test, y_pred, average='weighted')
    error_percentage = (1 - nb_acc) * 100

    # Create a new window to display the results
    result_window = tk.Toplevel(main)
    result_window.title("NaiveBayes Results")
    result_window.geometry("400x300")

    result_text = f'Accuracy for NaiveBayes is {nb_acc * 100:.2f}%\n'
    result_text += f'Precision for NaiveBayes is {nb_precision:.2f}\n'
    result_text += f'Recall for NaiveBayes is {nb_recall:.2f}\n'
    result_text += f'F1 Score for NaiveBayes is {nb_f1:.2f}\n'
    result_text += f'Error Percentage for NaiveBayes is {error_percentage:.2f}%\n'

    result_label = tk.Label(result_window, text=result_text, font=font1, justify=LEFT)
    result_label.pack(padx=20, pady=20)


naiveButton = tk.Button(main, text="Naive Bayes", command=NaiveBayes, bg="light blue", width=15)
naiveButton.place(x=220, y=700)
naiveButton.config(font=font1)


def SVM():
    global svm_acc, svm_precision, svm_recall, svm_f1
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred, average='weighted')
    svm_recall = recall_score(y_test, y_pred, average='weighted')
    svm_f1 = f1_score(y_test, y_pred, average='weighted')
    error_percentage = (1 - svm_acc) * 100

    # Create a new window to display the results
    result_window = tk.Toplevel(main)
    result_window.title("SVM Results")
    result_window.geometry("400x300")

    result_text = f'Accuracy for SVM is {svm_acc * 100:.2f}%\n'
    result_text += f'Precision for SVM is {svm_precision:.2f}\n'
    result_text += f'Recall for SVM is {svm_recall:.2f}\n'
    result_text += f'F1 Score for SVM is {svm_f1:.2f}\n'
    result_text += f'Error Percentage for SVM is {error_percentage:.2f}%\n'

    result_label = tk.Label(result_window, text=result_text, font=font1, justify=LEFT)
    result_label.pack(padx=20, pady=20)


svmButton = tk.Button(main, text="SVM", command=SVM, bg="light yellow", width=15)
svmButton.place(x=420, y=700)
svmButton.config(font=font1)


def RandomForest():
    global rf_acc, rf_precision, rf_recall, rf_f1, rf
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    rf_precision = precision_score(y_test, y_pred, average='weighted')
    rf_recall = recall_score(y_test, y_pred, average='weighted')
    rf_f1 = f1_score(y_test, y_pred, average='weighted')
    error_percentage = (1 - rf_acc) * 100

    # Create a new window to display the results
    result_window = tk.Toplevel(main)
    result_window.title("Random Forest Results")
    result_window.geometry("400x300")

    result_text = f'Accuracy for Random Forest is {rf_acc * 100:.2f}%\n'
    result_text += f'Precision for Random Forest is {rf_precision:.2f}\n'
    result_text += f'Recall for Random Forest is {rf_recall:.2f}\n'
    result_text += f'F1 Score for Random Forest is {rf_f1:.2f}\n'
    result_text += f'Error Percentage for Random Forest is {error_percentage:.2f}%\n'

    result_label = tk.Label(result_window, text=result_text, font=font1, justify=LEFT)
    result_label.pack(padx=20, pady=20)

rfButton = tk.Button(main, text="Random Forest", command=RandomForest, bg="light coral", width=15)
rfButton.place(x=620, y=700)
rfButton.config(font=font1)



import matplotlib.pyplot as plt

def accuracy_comparison():
    accuracies = {
        
        "Naive Bayes": nb_acc,
        "SVM": svm_acc,
        "Random Forest": rf_acc
    }

    algorithms = list(accuracies.keys())
    accuracy_values = [accuracies[alg] * 100 for alg in algorithms]  # Convert to percentage

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, accuracy_values, color=['blue', 'green', 'yellow'])
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison of Different Algorithms')
    plt.ylim(0, 100)

    for i in range(len(algorithms)):
        plt.text(i, accuracy_values[i] + 1, f"{accuracy_values[i]:.2f}%", ha='center')

    plt.show()

comparisonButton = tk.Button(main, text="Accuracy Comparison", command=accuracy_comparison, bg="light pink", width=20)
comparisonButton.place(x=820, y=700)
comparisonButton.config(font=font1)


def predict():
    global rf
    # Open file dialog to select the CSV file for prediction
    file_path = filedialog.askopenfilename(initialdir=".", title="Select CSV file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    
    # Load the data from the selected CSV file
    data = pd.read_csv(file_path)

    # Replace '?' with NaN
    data.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    data.fillna(data.mode().iloc[0], inplace=True)

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])

    # Predict using the trained Random Forest model
    predictions = rf.predict(data)

    # Display the predictions row-wise in the text box
    text.delete('1.0', END)
    text.insert(END, "Predictions:\n")
    for i, pred in enumerate(predictions, 1):
        text.insert(END, f"Quality of Row {i} values is: {pred}\n")

predictButton = tk.Button(main, text="Predict", command=predict, bg="light green", width=15)
predictButton.place(x=1120, y=700)
predictButton.config(font=font1)


main.config(bg='#32d1a7')
main.mainloop()
