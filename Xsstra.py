import random
import string
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

site_url = input("Enter site URL: ")
response = requests.get(site_url)
html_content = response.content
soup = BeautifulSoup(html_content, 'html.parser')
input_fields = soup.find_all('input')

dataset_file = 'xss_vulnerabilities.txt'

with open(dataset_file, 'r') as f:
    dataset = [{'input': line.strip(), 'output': 1} for line in f]

model = Sequential()
model.add(Dense(128, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputs = np.array([[len(item['input']), item['output']] for item in dataset])
outputs = np.array([item['output'] for item in dataset])

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.3)

model.fit(X_train, y_train, epochs=100, batch_size=64)

class WebPage:
    def __init__(self, input_fields):
        self.input_fields = input_fields
        self.successful_exploits = 0
        self.failed_exploits = 0
        self.attempts = 0
    
    def exploit(self, xss_payload):
        for input_field in self.input_fields:
            input_field_value = xss_payload
            input_field_name = input_field.get('name') or ''
            input_field_type = input_field.get('type') or ''
            input_field_value = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            if xss_payload in str(input_field):
                self.successful_exploits += 1
                return 1
        self.failed_exploits += 1
        return -1
    
    def reset(self):
        self.successful_exploits = 0
        self.failed_exploits = 0
        self.attempts = 0

webpage = WebPage(input_fields)

xss_payloads = []
for input_field in input_fields:
    state = np.array([0, 0])
    reward = 0
    xss_payload = ''
    while reward != 1 and webpage.attempts < 10:
        prob = model.predict(state)[0][0]
        if np.random.rand() < prob:
            char = random.choice(string.ascii_letters + string.digits)
            xss_payload = xss_payload + char
            state[0] = len(xss_payload)
        else:
            break
        reward = webpage.exploit(xss_payload)
        state[1] = reward
        webpage.attempts += 1
    xss_payloads.append(f"{input_field.get('name')}={xss_payload}'")

print("List of generated XSS payloads:")
print(xss_payloads)
