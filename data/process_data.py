import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pandas.io import sql

def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, on=['id'])

    return df


def clean_data(df):
    categories_df = df.categories.str.split(';', expand=True)
    row = categories_df.iloc[0]
    col_names = row.apply(lambda x: x[:-2]).tolist()
    categories_df.columns = col_names

    for col in col_names:
        categories_df[col] = pd.to_numeric(categories_df[col].astype(str).str[-1])

    # eliminate cols with single value
    numeric_cols = categories_df.select_dtypes([np.number]).columns
    stdev = categories_df[numeric_cols].std()
    drop_cols = stdev[stdev == 0].index
    categories_df.drop(columns=drop_cols, inplace=True)

    # replace 2's in related by 1's - TODO investigate further (does it improve ML algo performance?)
    categories_df.loc[categories_df.related == 2] = 1

    # drop old categories column, and insert new cat. cols
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories_df], axis=1)

    # drop duplicated instances
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = 'CategorizedMsgs'
    sql.execute(f'DROP TABLE IF EXISTS {table_name}', engine)
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()