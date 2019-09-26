import boto3

class SNS:

    def sendSMS(self, PhoneNumber, result):
        # Create an SNS client
        sns = boto3.client('sns')

        # Send a SMS message to the specified phone number
        response = sns.publish(
            PhoneNumber=PhoneNumber,
            Message=result 
        )

        # Print out the response
        print(response)

a = SNS()
a.sendSMS('0466636990', 'Hello Mohammed')