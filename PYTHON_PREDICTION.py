import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tkinter import *
from PIL import ImageTk,Image
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import datasets
import statsmodels.api
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from matplotlib.figure import Figure

#reading datafile and taining the models

# loading the dataset to pandas Dataform

loan_dataset = pd.read_csv('train.csv')

# deleting the missing values
loan_dataset = loan_dataset.dropna()

#replacing loan status with 0/1
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# replacing value,of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace = '3+', value=4)

# converting categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'Yes':1,'No':0},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

#seprating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status','LoanAmount','Loan_Amount_Term',],axis=1)
Y = loan_dataset['Loan_Status']

#spliting data into training data and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

#SPORT VECTOR MACHIN MODEL loading model and fiting data
model1=svm.SVC(kernel='linear')
model1.fit(X_train,Y_train)

#Multipal linear regration based on creadit_histry, married and education status and property area
model2=linear_model.LinearRegression()
model2.fit(loan_dataset[["Credit_History","Married","Education","Property_Area"]],loan_dataset["Loan_Status"])

#Random Forest Regression
model3 = RandomForestRegressor(n_estimators=100, max_depth=5)
model3.fit(X_train, Y_train)

# Decision Tree Regression
model4 = DecisionTreeRegressor(max_depth=5)
model4.fit(X_train, Y_train)

# Lasso Regression
model5 = Lasso(alpha=0.1)
model5.fit(X_train, Y_train)

# for loan amount 

#seprating the data and label

XX = loan_dataset.drop(columns=['Loan_ID','Loan_Status','Credit_History','Property_Area','LoanAmount','Loan_Amount_Term'],axis=1)
YY = loan_dataset['LoanAmount']

XX_train,XX_test,YY_train,YY_test = train_test_split(XX,YY,test_size=0.1)

# #SPORT VECTOR MACHIN MODEL loading model and fiting data
# Amodel1=svm.SVC(kernel='linear')
# Amodel1.fit(XX_train,YY_train)

#Multipal linear regration based on creadit_histry, married and education status and property area
Amodel2=linear_model.LinearRegression()
Amodel2.fit(XX_train,YY_train)

#Random Forest Regression
Amodel3 = RandomForestRegressor(n_estimators=100, max_depth=5)
Amodel3.fit(XX_train,YY_train)

# Decision Tree Regression
Amodel4 = DecisionTreeRegressor(max_depth=5)
Amodel4.fit(XX_train,YY_train)

# Lasso Regression
Amodel5 = Lasso(alpha=0.1)
Amodel5.fit(XX_train,YY_train)


window=Tk()
window.title("Loan and Ammount Prediction")
window.geometry('500x600')
window.resizable(False, False)

frame = Frame(window, width=480, height=580, bg='#9898F5')
frame.place(x=10,y=10)

#declarind varabals for input

loan_id=StringVar()
gender=StringVar()
married=StringVar()
dependents=IntVar()
educated=BooleanVar()
self_emp=BooleanVar()
totalInc=DoubleVar()
otherInc=DoubleVar()
credit=StringVar()
area=StringVar()

# creatin lable for input used in loan

lable1=Label(window,text='Enter Loan ID : ',fg='black',font=('Arial',10), bg='#9898F5')
lable1.grid(row=0,column=0,padx=5,pady=20,sticky='e')

lable2=Label(window,text='Gender(M/F) : ',fg='black',font=('Arial',10), bg='#9898F5')
lable2.grid(row=1,column=0,padx=5,pady=10,sticky='e')

lable3=Label(window,text='Married(Y/N)  : ',fg='black',font=('Arial',10), bg='#9898F5')
lable3.grid(row=2,column=0,padx=5,pady=10,sticky='e')

lable4=Label(window,text='Depenents(0/1/2/3/4)  : ',fg='black',font=('Arial',10), bg='#9898F5')
lable4.grid(row=3,column=0,padx=5,pady=10,sticky='e')

lable5=Label(window,text='Educated  : ',fg='black',font=('Arial',10), bg='#9898F5')
lable5.grid(row=4,column=0,padx=5,pady=10,sticky='e')

lable6=Label(window,text='Self-Employed : ',fg='black',font=('Arial',10), bg='#9898F5')
lable6.grid(row=5,column=0,padx=5,pady=10,sticky='e')

lable7=Label(window,text='Applicant total Income : ',fg='black',font=('Arial',10), bg='#9898F5')
lable7.grid(row=6,column=0,padx=5,pady=10,sticky='e')

lable8=Label(window,text='Income from other sorce : ',fg='black',font=('Arial',10), bg='#9898F5')
lable8.grid(row=7,column=0,padx=5,pady=10,sticky='e')

lable9=Label(window,text='Credit_History(Y/N) : ',fg='black',font=('Arial',10), bg='#9898F5')
lable9.grid(row=8,column=0,padx=5,pady=10,sticky='e')

lable10=Label(window,text='Property Area(rural/urban/semiurbon): ',fg='black',font=('Arial',10), bg='#9898F5')
lable10.grid(row=9,column=0,padx=5,pady=10,sticky='e')

#taking input box

textbox1=Entry(window,textvariable=loan_id,fg='black',font=('Arial',10))
textbox1.grid(row=0,column=1,sticky='e')

textbox2=Entry(window,textvariable=gender,fg='black',font=('Arial',10))
textbox2.grid(row=1,column=1,sticky='e')

textbox3=Entry(window,textvariable=married,fg='black',font=('Arial',10))
textbox3.grid(row=2,column=1,sticky='e')

textbox4=Entry(window,textvariable=dependents,fg='black',font=('Arial',10))
textbox4.grid(row=3,column=1,sticky='e')

checkbox1 = Checkbutton(window, text="",variable=educated, bg='#9898F5')
checkbox1.grid(row=4,column=1)

checkbox2 = Checkbutton(window, text="",variable=self_emp, bg='#9898F5')
checkbox2.grid(row=5,column=1)

textbox5=Entry(window,textvariable=totalInc,fg='black',font=('Arial',10))
textbox5.grid(row=6,column=1,sticky='e')

textbox6=Entry(window,textvariable=otherInc,fg='black',font=('Arial',10))
textbox6.grid(row=7,column=1,sticky='e')

textbox7=Entry(window,textvariable=credit,fg='black',font=('Arial',10))
textbox7.grid(row=8,column=1,sticky='e')

textbox8=Entry(window,textvariable=area,fg='black',font=('Arial',10))
textbox8.grid(row=9,column=1,sticky='e')

def predict_vs_actual_data(actual_data,predicted_data,dataframe,color="red"):
    #sns plot
    sns.lmplot(x=actual_data, y=predicted_data, data=dataframe, fit_reg=False,height=5,aspect=2)
    #lining the data
    d_line= np.arange(dataframe.min().min(), dataframe.max().max())
    #ploting 
    plt.plot(d_line, d_line, color=color, linestyle='--')
    plt.show()


#prediction funcrion 

def plot_amodel2():
   X_test_pred = Amodel2.predict(XX_test)
   Amodel2_dataframe= pd.DataFrame({"actual_data":YY_test,"predicted_data":X_test_pred})
   predict_vs_actual_data(actual_data='actual_data',predicted_data='predicted_data',dataframe=Amodel2_dataframe)

def plot_amodel3():
   X_test_pred = Amodel3.predict(XX_test)
   Amodel3_dataframe= pd.DataFrame({"actual_data":YY_test,"predicted_data":X_test_pred})
   predict_vs_actual_data(actual_data='actual_data',predicted_data='predicted_data',dataframe=Amodel3_dataframe)

def plot_amodel4():
   X_test_pred = Amodel4.predict(XX_test)
   Amodel4_dataframe= pd.DataFrame({"actual_data":YY_test,"predicted_data":X_test_pred})
   predict_vs_actual_data(actual_data='actual_data',predicted_data='predicted_data',dataframe=Amodel4_dataframe)

def plot_amodel5():
   X_test_pred = Amodel5.predict(XX_test)
   Amodel5_dataframe= pd.DataFrame({"actual_data":YY_test,"predicted_data":X_test_pred})
   predict_vs_actual_data(actual_data='actual_data',predicted_data='predicted_data',dataframe=Amodel5_dataframe)





#MAKING WINDOW FOR GRAPH  ANALYSIS

def analysis():

    top=Toplevel()
    top.geometry("400x355")

    frame1 = Frame(top, width=390, height=340, bg='#9898F5')
    frame1.place(x=5,y=5)
    top.resizable(False, False)
    ltop1=Label(top,text='GRAPH --> MULTIPAL LINEAR REGRAION ',fg='black',font=('Arial',10), bg='#9898F5')
    ltop1.grid(row=0,column=0,padx=5,pady=10)

    ltop2=Label(top,text='GRAPH --> RANDOM FOREST REGRAION ',fg='black',font=('Arial',10), bg='#9898F5')
    ltop2.grid(row=1,column=0,padx=5,pady=10)

    ltop3=Label(top,text='GRAPH --> DECISION TREE REGRAION ',fg='black',font=('Arial',10), bg='#9898F5')
    ltop3.grid(row=2,column=0,padx=5,pady=10)

    ltop4=Label(top,text='GRAPH --> LASSO REGRAION ',fg='black',font=('Arial',10), bg='#9898F5')
    ltop4.grid(row=3,column=0,padx=5,pady=10)


    graph1=Button(top,command=plot_amodel2,text='Graph',fg='black',font=('Arial',10))
    graph1.grid(row=0,column=1,pady=10)

    graph2=Button(top,command=plot_amodel3,text='Graph',fg='black',font=('Arial',10))
    graph2.grid(row=1,column=1,pady=10)

    graph3=Button(top,command=plot_amodel4,text='Graph',fg='black',font=('Arial',10))
    graph3.grid(row=2,column=1,pady=10)

    graph4=Button(top,command=plot_amodel5,text='Graph',fg='black',font=('Arial',10))
    graph4.grid(row=3,column=1,pady=10)

    Amodel2_accuracy_test = round(abs(Amodel2.score(XX_test,YY_test)),2)
    Amodel3_accuracy_test = round(abs(Amodel3.score(XX_test,YY_test)),2)
    Amodel4_accuracy_test = round(abs(Amodel4.score(XX_test,YY_test)),2)
    Amodel5_accuracy_test = round(abs(Amodel5.score(XX_test,YY_test)),2)

    ltop5=Label(top,text='ACCURACY --> MULTIPAL LINEAR REGRAION :: ' + str(Amodel2_accuracy_test),fg='black',font=('Arial',10), bg='#9898F5')
    ltop5.grid(row=4,column=0,padx=5,pady=10,sticky='e')

    ltop6=Label(top,text='ACCURACY --> RANDOM FOREST REGRAION :: ' + str(Amodel3_accuracy_test),fg='black',font=('Arial',10), bg='#9898F5')
    ltop6.grid(row=5,column=0,padx=5,pady=10,sticky='e')

    ltop7=Label(top,text='ACCURACY --> DECISION TREE REGRAION ::' + str(Amodel4_accuracy_test),fg='black',font=('Arial',10), bg='#9898F5')
    ltop7.grid(row=6,column=0,padx=5,pady=10,sticky='e')

    ltop8=Label(top,text='ACCURACY --> LASSO REGRAION ::' + str(Amodel5_accuracy_test),fg='black',font=('Arial',10), bg='#9898F5')
    ltop8.grid(row=7,column=0,padx=5,pady=8,sticky='e')



def Predict_staust():
    prelist=[]
    loanA=[]
    if(gender.get()=='M' or gender.get=='m'):
        prelist.append(1)
        loanA.append(1)
    else:
        prelist.append(0)
        loanA.append(0)

    if(married.get()=='Y' or married.get=='y'):
        prelist.append(1)
        loanA.append(1)
    else:
        prelist.append(0)
        loanA.append(0)

    prelist.append(dependents.get())
    loanA.append(dependents.get())
    if(educated.get()==TRUE):
        prelist.append(1)
        loanA.append(1)
    else:
        prelist.append(0)
        loanA.append(0)

    if(self_emp.get()==TRUE):
        prelist.append(1)
        loanA.append(1)
    else:
        prelist.append(0)
        loanA.append(0)    

    prelist.append(totalInc.get())
    prelist.append(otherInc.get())
    loanA.append(totalInc.get())
    loanA.append(otherInc.get())

    if(credit.get()=='Y' or credit.get=='y'):
        prelist.append(1)
    else:
        prelist.append(0)

    if(area.get().lower=='rural'):
        prelist.append(0)
    elif(area.get().lower=='urban'):
        prelist.append(2)
    else:
        prelist.append(1)

    linearlist=[]
    if(credit.get()=='Y' or credit.get=='y'):
        linearlist.append(1)
    else:
        linearlist.append(0)

    if(married.get()=='Y' or married.get=='y'):
        linearlist.append(1)
    else:
        linearlist.append(0)
    
    if(educated.get()==TRUE):
        linearlist.append(1)
    else:
        linearlist.append(0)
    
    if(area.get().lower=='rural'):
        linearlist.append(0)
    elif(area.get().lower=='urban'):
        linearlist.append(2)
    else:
        linearlist.append(1)

    mainlist1=[]
    mainlist1.append(prelist)
    mainlin=[]
    mainlin.append(linearlist)
    ammontlist=[]
    ammontlist.append(loanA)
    #prediction

    p1=model1.predict(mainlist1)
    p2=model2.predict(mainlin)
    p3=model3.predict(mainlist1)
    p4=model4.predict(mainlist1)
    p5=model5.predict(mainlist1)

    a2=Amodel2.predict(ammontlist)
    a3=Amodel3.predict(ammontlist)
    a4=Amodel4.predict(ammontlist)
    a5=Amodel5.predict(ammontlist)

    # print("PREDCTION : ",a2,a3,a4,a5)
    # print(p1,p2,p3,p4,p5)

    predstatus = (p1[0]+p2[0])/2
    if(predstatus > 0.5 ):
        emptylable.config(text='Loan can be Granted !')
        loan_amount= (a2[0]+a3[0]+a4[0]+a5[0])/4
        loan_amount=round(loan_amount, 2)
        emptylable1.config(text='Granted Loan Amount is  : ' + str(loan_amount))
    else:
        emptylable.config(text='')
        emptylable1.config(text='We can not grant Loan !')
    

    
# submit button

button1=Button(window,command=Predict_staust,text='Save',fg='black',font=('Arial',10))
button1.grid(row=10,column=1,pady=10)

button2=Button(window,command=analysis,text='Analyize',fg='black',font=('Arial',10))
button2.grid(row=11,column=1,pady=10)


emptylable=Label(window,fg='black',font=('Arial',10), bg='#9898F5')
emptylable.grid(row=12,column=1,sticky='w')

emptylable1=Label(window,fg='black',font=('Arial',10), bg='#9898F5')
emptylable1.grid(row=13,column=1,sticky='w')


window.mainloop()