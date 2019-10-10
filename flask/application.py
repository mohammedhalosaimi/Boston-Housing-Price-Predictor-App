
from modelling import Modelling
from sns import Sns
from flask import Flask, request, render_template
import json

application = Flask(__name__)

@application.route('/')
# @application.route('/')
def getFlaks():

        # columns values to allow for user selection
        model = Modelling()
        df, user_input_list = model.data_preparation()

        return render_template('main.html', selections=user_input_list)




@application.route('/prediction', methods=['POST'])
# @application.route('/')
def postFlaks():

    # get user's inputs
    user_CRIM = request.form.get("CRIM", False)
    user_ZN = request.form.get("ZN", False)
    user_INDUS = request.form.get("INDUS", False)
    user_CHAS = request.form.get("CHAS", False)
    user_NOX = request.form.get("NOX", False)
    user_RM = request.form.get("RM", False)
    user_AGE = request.form.get("AGE", False)
    user_DIS = request.form.get("DIS", False)
    user_RAD = request.form.get("RAD", False)
    user_TAX = request.form.get("TAX", False)
    user_PTRATIO = request.form.get("PTRATIO", False)
    user_B = request.form.get("B", False)
    user_LSTAT = request.form.get("LSTAT", False)

    
    # get user's phone nuber
    user_phone_number = request.form.get("phone", False)       

    # instantiate the model
    model = Modelling()

    prediction = model.predictUserInput(user_CRIM, user_ZN, user_INDUS, user_CHAS,
    user_NOX, user_RM, user_AGE, user_DIS, user_RAD, user_TAX, user_PTRATIO, user_B, user_LSTAT)
    
    # append meaniningful message to the prediction
    prediction = "Predicted Median value of owner-occupied homes in $1000's is: " + prediction
   
    # if the phone number is valid, then send a message
    snsService = Sns()
    # user_phone_number = str(user_phone_number)
    returnMessage = snsService.sendSMS(user_phone_number, prediction)

    # check if user did not provide phone nubmer, then don't return any message from the SNS class
    if len(user_phone_number) == 0:
        returnMessage = ""
    
    # prediction = 'RESULT'
    return render_template('prediction.html', result=[prediction, returnMessage])



if __name__ == '__main__':
    application.run()
