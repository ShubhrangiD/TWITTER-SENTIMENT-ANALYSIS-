# Twitter Sentiment Analysis

This project demonstrates how to perform sentiment analysis on a Twitter dataset using Python and various machine learning libraries. The dataset is fetched from Kaggle, and we use logistic regression to classify tweets as positive or negative.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- pip

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ShubhrangiD/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the Kaggle library:
    ```bash
    pip install kaggle
    ```

4. Configure the Kaggle API key:
    - Place your `kaggle.json` file in the root directory of the project.
    - Configure the path:
      ```bash
      mkdir -p ~/.kaggle
      cp kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json
      ```

## Dataset

The dataset used is the Sentiment140 dataset from Kaggle. To fetch the dataset, run:
```bash
kaggle datasets download -d kazanova/sentiment140
```

Extract the downloaded dataset:
```python
from zipfile import ZipFile
dataset = 'sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
```

## Data Processing

1. Load the data:
    ```python
    import pandas as pd

    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')
    ```

2. Process the data:
    ```python
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    def stemming(content):
        pattern = re.compile('[^a-zA-Z]')
        stop_words = set(stopwords.words('english'))
        port_stem = PorterStemmer()

        stemmed_content = pattern.sub(' ', content)
        stemmed_content = stemmed_content.lower().split()
        stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
        return ' '.join(stemmed_content)

    twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)
    ```

3. Split the data into training and test sets:
    ```python
    from sklearn.model_selection import train_test_split

    X = twitter_data['stemmed_content'].values
    Y = twitter_data['target'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    ```

## Model Training

1. Convert textual data to numerical data:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    ```

2. Train the logistic regression model:
    ```python
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    ```

## Model Evaluation

Evaluate the model using accuracy score:
```python
from sklearn.metrics import accuracy_score

# On training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data:', training_data_accuracy)

# On test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the testing data:', testing_data_accuracy)
```

## Saving and Loading the Model

Save the trained model for future use:
```python
import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

Load the saved model:
```python
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
```

## Usage

To use the saved model for predictions:
```python
X_new = X_test[200]
prediction = loaded_model.predict([X_new])
print(prediction)
```

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
