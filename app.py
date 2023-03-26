#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
import pyrebase
from flask import Flask,jsonify,request
import json
import pandas as pd # dataframe
import numpy as np # mathematical calculations
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor #Random Forest Regression
from sklearn.model_selection import train_test_split
from warnings import simplefilter #Filtering warnings
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
from matplotlib import rcParams
import smtplib
from email.message import EmailMessage


app = Flask(__name__) #intance of our flask application 

@app.route('/detect', methods = ['GET','POST'])
def detect():
    global response
    if(request.method=='POST'):
        request_data=request.data
        request_data=json.loads(request_data.decode('utf-8'))

        model=request_data['model']
        year=request_data['year']
        engineCapacity=request_data['engine capacity']
        mileage=request_data['mileage']
        grade=request_data['grade']
        fuelType=request_data['fuel type']
        transType=request_data['transmission type']

        simplefilter(action='ignore', category=FutureWarning) 

        # importing the dataset
        df1 = pd.read_csv('toyota_cars.csv')
        df1.rename(columns = {'price (Rs.)' : 'price'}, inplace = True)

        df2 = df1.drop('price', axis = 'columns')
        df3 = df1.drop(['make', 'model', 'model_year', 'gear', 'fuel_type', 'engine_cc','mileage_km'], axis = 'columns')

        make_l = LabelEncoder()
        model_l = LabelEncoder()
        gear_l = LabelEncoder()
        fuel_type_l = LabelEncoder()

        df2['make_n'] = make_l.fit_transform(df2['make'])
        df2['model_n'] = model_l.fit_transform(df2['model'])
        df2['transmission_n'] = gear_l.fit_transform(df2['gear'])
        df2['fuel_type_n'] = fuel_type_l.fit_transform(df2['fuel_type'])

        modelNum=findMatchNum(model,"model","model_n",df2)
        transNum=findMatchNum(transType,"gear","transmission_n",df2)
        fuelNum=findMatchNum(fuelType,"fuel_type","fuel_type_n",df2)

        # input data 
        x = df2.drop(['make', 'model', 'gear', 'fuel_type'], axis = 'columns')

        # output data
        y = df3

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
        model = RandomForestRegressor()
        model.fit(x_train,y_train)
        model.fit(x,y)

        prediction = model.predict([[int(year),int(engineCapacity),int(mileage), 0, int(modelNum), int(transNum), int(fuelNum)]]).astype(float)
        price = "Rs. {:,.2f}".format(prediction[0])



        return jsonify({'response':price})
    else:
        return jsonify({'response':response})

# @app.route('/graph',methods=['POST'])
# def graph():
#     request_data=request.data
#     request_data=json.loads(request_data.decode('utf-8'))

#     model=request_data['model']
#     email=request_data['email']
    

#     # importing the dataset
#     df1 = pd.read_csv('toyota_cars.csv')
#     df1.rename(columns = {'price (Rs.)' : 'price'}, inplace = True)

#     df2 = df1.drop('price', axis = 'columns')
#     df3 = df1.drop(['make', 'model', 'model_year', 'gear', 'fuel_type', 'engine_cc','mileage_km'], axis = 'columns')

#     # dropped the unnecessary columns
#     # get details of a certain model
#     df4 = df1.drop(['make', 'fuel_type', 'engine_cc'], axis = 'columns')
#     df4 = df4[df4['model'] == model]

#     price_fluctuation = df4.groupby('model_year')['price'].mean().astype(int)

#     # inialise the plot  
#     fig = plt.figure(figsize=(15,10))
#     # (111) -> number of rows=1, number of columns=1, plot number=1
#     ax = fig.add_subplot(111)

#     # plot the data
#     # x=year(index of the price_fluctuation table), y=price
#     ax.plot(price_fluctuation.index, price_fluctuation)

#     ax.set_title('Price fluctuation over the years')
#     ax.set_xlabel('Year')
#     ax.set_ylabel('Price')

#     # saving the plot as a jpg file
#     # removed the whitespace around the figure
#     plt.savefig('graph.jpg', bbox_inches='tight')
#     #plt.show()

#     split_email=email.split('@')
    

#     storage.child("graphs/"+split_email[0]+".jpg").put("graph.jpg")
#     url=storage.child("graphs/"+split_email[0]+".jpg").get_url(None)


#     return jsonify({'response':url})


#     request_data=request.data
#     request_data=json.loads(request_data.decode('utf-8'))

#     email=request_data['email']
#     newPassword=request_data['new password']


#     user=adminAuth.get_user_by_email(email)
#     adminAuth.update_user(user.uid,password=newPassword)


#     return jsonify({'response':'Password changed successfully'})

# @app.route('/saveDetails',methods=['POST'])
# def saveDetails():
#     global detailsChanged
#     request_data=request.data
#     request_data=json.loads(request_data.decode('utf-8'))

#     email=request_data['email']
#     username=request_data['username']

#     newEmail=request_data['new email']
#     newUsername=request_data['new username']


#     user=adminAuth.get_user_by_email(email)
#     if(newEmail!=email or newUsername!=username):
#         adminAuth.update_user(user.uid,email=newEmail,display_name=newUsername)
#         detailsChanged="true"

    
    

#     return jsonify({'response':detailsChanged})


# @app.route('/feedback',methods=['POST'])
# def feedback():
#     request_data=request.data
#     request_data=json.loads(request_data.decode('utf-8'))

#     username=request_data['username']
#     email=request_data['email']
#     userMessage=request_data['message']


#     msg=EmailMessage();
#     msg['Subject']="message from autopredict user, "+str(email)
#     msg['From']="user"
#     msg['To']="autopredict2021@gmail.com"



#     body=str(userMessage)

#     message=f"""Hello AutoPredict,

#     {body}

#     Thank You,
#     {username}"""

#     msg.set_content(message)



#     server=smtplib.SMTP_SSL('smtp.gmail.com',465)
#     server.login("autopredict2021@gmail.com","autopredict123")
#     server.send_message(msg)
#     server.quit()

#     return jsonify({'response':'message sent...'})


# @app.route('/signin',methods=['POST'])
# def signIn():
#     request_data=request.data
#     request_data=json.loads(request_data.decode('utf-8'))

#     email=request_data['email']
#     password=request_data['password']

#     validation=""
#     try:
#         user=auth.sign_in_with_email_and_password(email,password)
#         validation="valid"
#     except:
#         validation="invalid"
    
#     user=adminAuth.get_user_by_email(email)
    

#     return jsonify({'response':validation,'username':user.display_name,'email':user.email,'photoURL':user.photo_url})


# @app.route('/register',methods=['GET','POST'])
# def register():
    # if(request.method=='POST'):
    #     request_data=request.data
    #     request_data=json.loads(request_data.decode('utf-8'))

    #     userName=request_data['username']
    #     email=request_data['email']
    #     password=request_data['password']

    #     #create user
    #     user=adminAuth.create_user(
    #         email=email,
    #         password=password,
    #         display_name=userName)
    #     #auth.send_email_verification(user['idToken'])

        
    #     return jsonify({'response':'successfully user has been created'})


if __name__ == "__main__":
    app.run(debug = True) #debug will allow changes without shutting down the server 

