
from modelling import Modelling
from sns import SNS
from flask import Flask, request, render_template
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
# @app.route('/')
def hello():
    if request.method == 'GET':

        # columns values to allow for user selection
        model = Modelling()
        df, user_input_list = model.data_preparation()

        print("I am in flask")
        return render_template('hello.html', selections=user_input_list)

    # # post to the user
    elif request.method == 'POST':

        # get values from user
        user_Departing_Port = request.form['Departing_Port']
        user_Arriving_Port = request.form['Arriving_Port']
        user_Airline = request.form['Airline']
        user_Sectors_Scheduled = request.form['Sectors_Scheduled']
        user_Sectors_Flown = request.form['Sectors_Flown']
        user_Year = request.form['Year']
        user_Month_Num = request.form['Month_Num']

        user_phone_number = request.form['phone']
        user_email_address = request.form['email']

        
        prediction = model.predictUserInput(Departing_Port, Arriving_Port, Airline, Month_Num)

        if len(user_phone_number) > 10:
            SNS.sendSMS(user_phone_number, prediction)

            
        return render_template('hello.html', result=prediction)



if __name__ == '__main__':
    app.run()
