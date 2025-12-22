from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

def open_new_window():

    color = "#BDCDD6"
    root1 = tk.Toplevel()
    root1.title('Intelligent Hepatitis Prediction System (IHPS)')
    root1.geometry('1200x600+900+200')
    root1.configure(background=color)
    root1.resizable(False, False)

    img = Image.open("Rounded Rectangle 1.png")
    img_resized = img.resize((200, 150))
    box = ImageTk.PhotoImage(img_resized)
    Label(root1, image=box, bg='#203243').place(x=850, y=90, width=300, height=280)

    labels = ["age", "gender", "steroid", "antivirals", "fatigue",
              "malaise", "anorexia", "liverBig", "liverFirm", "spleen"]

    labels2 = ["spiders", "ascites", "varices", "bili",
               "alk", "sgot", "albu", "protime", "histology"]

    entries = []
    clean_data = pd.read_csv("clean_data")
    X = clean_data[['age', 'gender', 'steroid', 'antivirals', 'fatigue',
                   'malaise', 'anorexia', 'liverBig', 'liverFirm', 'spleen', 'spiders',
                   'ascites', 'varices', 'bili', 'alk', 'sgot', 'albu', 'protime',
                   'histology']]

    y = clean_data['target'].replace({1: 0, 2: 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    nb.fit(X_train, y_train)


    for i, label_text in enumerate(labels):
        tk.Label(root1, text=label_text + ':', font=('Arial', 15), bg=color).place(x=30, y=50 + i * 50)
        var = tk.StringVar()
        tk.Entry(root1, bd=4, font=('Arial', 16), textvariable=var).place(x=210, y=50 + i * 50, width=200, height=30)
        entries.append(var)

    for j, label_text_j in enumerate(labels2):
        tk.Label(root1, text=label_text_j + ':', font=('Arial', 15), bg=color).place(x=450, y=50 + j * 50)
        var2 = tk.StringVar()
        tk.Entry(root1, bd=4, font=('Arial', 16), textvariable=var2).place(x=630, y=50 + j * 50, width=200, height=30)
        entries.append(var2)
    def print_values(entries):

        values = [float(var.get()) for var in entries]
        new_patient = np.array(values).reshape(1, -1)

        pred_nb = nb.predict(new_patient)

        print("Entered values (int):", values)
        print("Random Forest Prediction:", pred_nb[0])

        if pred_nb == 0:
            img = Image.open("0.png")
            img = img.resize((200, 150))
            img_tk = ImageTk.PhotoImage(img)

            lbl = tk.Label(root1, image=img_tk, bg='#203243')
            lbl.image = img_tk
            lbl.place(x=850, y=120, width=300, height=200)
            IS_PATIENT= Label(root1, text='Patient status: Healthy', font=(HORIZONTAL, 14), bg='#203243',
                         fg="GREEN")
            IS_PATIENT.place(x=900, y=330)

        else:
            img = Image.open("1 (2).png")
            img = img.resize((200, 150))
            img_tk = ImageTk.PhotoImage(img)

            lbl = tk.Label(root1, image=img_tk, bg='#203243')
            lbl.image = img_tk
            lbl.place(x=850, y=120, width=300, height=200)
            IS_SICK = Label(root1, text='Patient status:Sick', font=(HORIZONTAL, 14), bg='#203243',
                               fg="red")
            IS_SICK.place(x=900, y=330)


    tk.Button(
        root1,
        text='Prediction',
        font=('Arial', 18),
        bg='#D8D8D8',
        bd=3,
        command=lambda: print_values(entries)
    ).place(x=630, y=500, width=200, height=40)


    root1.mainloop()


def Connecte():
    nom_user = entry1.get()
    MP = entry2.get()
    if nom_user == "" or MP == "":
        messagebox.showerror("", 'Error: username or password empty')
        entry1.delete(0, END)
        entry2.delete(0, END)
    elif nom_user == "IA" and MP == "1234IA":
        open_new_window()
    else:
        messagebox.showwarning("", "Try again")
        entry1.delete(0, END)
        entry2.delete(0, END)



root0=Tk()

root0.title('Intelligent Hepatitis Prediction System (IHPS)')
root0.geometry('1200x600+450+200')
root0.configure(bg='#6096B4')
root0.resizable(False,False)
#background image
frame0=Frame(root0,bg='red')
frame0.pack(fill=Y)
backgroundimage=PhotoImage(file='imgAPI (2).png')
#label(frame0,image=backgroundimage).pack()
background_label = tk.Label(root0, image=backgroundimage)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

frame00 = Frame(root0,width=400,height=300,bg='#43A047',borderwidth=5)
frame00.place(x=400 , y=100)

lbltitre=Label(frame00 , borderwidth=3 , relief=SUNKEN
               ,text='IHPS' ,font=('Sans Serif',25),bg='#203243',fg='white')
lbltitre.place(x=0,y=0,width=400)

user=Label(frame00, text='Username:',font=('Arial',15),fg='black',bg='#43A047')
user.place(x=30,y=90)

entry1 = Entry(frame00,bd=4,font=('Arial',16))
entry1.place(x=150,y=90,width=200,height=30)

PW=Label(frame00,text='Passworde:',font=('Arial',16),fg='black',bg='#43A047')
PW.place(x=30,y=150)

entry2= Entry(frame00,show="*",bd=4,font=('Arial',16))
entry2.place(x=150,y=150,width=200,height=30)

connexion=Button(frame00 , text='connexion',font=('Arial',17),bg='#256D85', highlightthickness=0,fg='white',command=Connecte)
connexion.place(x=150,y=200,width=200)

root0.mainloop()

