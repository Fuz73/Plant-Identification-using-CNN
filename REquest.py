from google.colab import files
uploaded = files.upload()


import requests

url = '<APP_PATH>'  # Replace with your ngrok URL
files = {'file': open('<IMAGE_PATH>', 'rb')}  # Use the name of the uploaded file

response = requests.post(url, files=files)

# Check the response status code
if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)  # Print error details for debugging
