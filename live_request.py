import requests

inputs = {'age': 48, 'workclass': 'Private', 'fnlgt': 45612,
          'education': 'HS-grad',
          ' education-num': 9, ' marital-status': 'Never-married',
          'occupation': 'Adm-clerical',
          'relationship': 'Unmarried', 'race': 'Black',
          'sex': 'Female', ' capital-gain': 0, ' capital-loss': 0,
          ' hours-per-week': 37, ' native-country': 'United-States'}

url = 'https://udactiy-api.onrender.com/predict'
x = requests.post(url, json=inputs, verify=True)
print(x.text)
print(x.status_code)
