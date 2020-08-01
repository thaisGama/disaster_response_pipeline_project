import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentences = sent_tokenize(text)

        for sentence in sentences:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags

            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class ModelSelector(BaseEstimator):
    def __init__(self, classifier: str = 'RandomForestClassifier'):
        self.classifier = classifier

    def fit(self, X, y=None):
        if self.classifier == 'RandomForestClassifier':
            self.classifier_ = RandomForestClassifier()
        elif self.classifier == 'ExtraTreesClassifier':
            self.classifier_ = ExtraTreesClassifier
        else:
            raise ValueError(
                'Unknown Classifier. Allowed classifiers are RandomForestClassifier or ExtraTreesClassifier')
        self.classifier_.fit(X, y)

    def predict(self, X, y=None):
        return self.classifier_.predict(X)


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')

    df = pd.read_sql('SELECT * FROM CategorizedMsgs', con=engine)

    x = df.message
    y = df.drop(['message', 'categories', 'id', 'original', 'genre'], inplace=True)

    return x, y


def tokenize(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    # TODO check if lemmatizer/CountVectorizer remove stopwords!!!
    return clean_tokens


def build_model():
    pipeline = Pipeline([('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())
    ])),
                         ('clf', ModelSelector())

                         ])


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

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


if __name__ == '__main__':
    main()
