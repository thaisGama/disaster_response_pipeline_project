import sys
import pandas as pd
from sqlalchemy import create_engine
import time

import re
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import SelectFromModel

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
not_text_regex = '[^a-zA-Z0-9]'


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        # TODO figure out why wrong pos_tag [] sentence: what should i do
        sentence_list = nltk.sent_tokenize(text)

        regexTokenizer = RegexpTokenizer(r'\w+')

        for sentence in sentence_list:
            sentence = re.sub(not_text_regex, ' ', sentence).strip()
            if not regexTokenizer.tokenize(sentence.strip()):
                return False

            pos_tags = nltk.pos_tag(tokenize(sentence.strip()))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class ModelSelector(BaseEstimator):
    def __init__(
            self,
            estimator=RandomForestClassifier(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')

    df = pd.read_sql('SELECT * FROM CategorizedMsgs', con=engine)
    x = df.message
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(y)

    return x, y, category_names


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    text = re.sub(not_text_regex, ' ', text)
    tokens = word_tokenize(text)

    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stop_words]
    return clean_tokens


def build_model():
    pipeline = Pipeline([('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())
    ])),
                         ('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=100))),
                         ('clf', MultiOutputClassifier(ModelSelector()))
                         ])
    print(pipeline.get_params())

    parameters = [
        {'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
            #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
            'clf__estimator__estimator': [LinearSVC()],
        },
        {
            'clf__estimator__estimator': [RandomForestClassifier()],
            'clf__estimator__estimator__class_weight': ['balanced', 'balanced_subsample'],
            'clf__estimator__estimator__n_estimators': [1, 10, 100, 1000],
            'clf__estimator__estimator__n_jobs': [-1],
        },
        {
            'clf__estimator__estimator': [ExtraTreesClassifier()],
            'clf__estimator__estimator__class_weight': ['balanced', 'balanced_subsample'],
            'clf__estimator__estimator__n_estimators': [10, 100, 1000],
            'clf__estimator__estimator__n_jobs': [-1]
    }
    ]
    cv_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=5)
    return cv_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print("\nBest Parameters:", model.best_params_)
    df_pred = pd.DataFrame(y_pred, columns=category_names)

    for col in category_names:
        print(classification_report(Y_test[col].values, df_pred[col].values))



def save_model(model, model_filepath):
    outfile = open(model_filepath, 'wb')
    pickle.dump(model, outfile)
    outfile.close()


def main():
    start = time.time()
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_train = time.time()
        model.fit(X_train, Y_train)
        print(f'Build model time minutes: {(time.time() - start_train)/60}')

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    print(f"total time minutes: {(time.time() - start)/60}")


if __name__ == '__main__':
    main()
