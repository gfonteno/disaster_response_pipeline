import sys

# - Import Python libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """ 
    input:
    	messages_filepath: messages dataset file location and name
    	categories_filepath: categories dataset file location and name
    	        
    output:
		1. read two datasets
		2. merge both datasets
		3. return pandas dataset 'df'
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    # return the dataframe 
    return df


def clean_data(df):

    """ 
    input:
    	df: dataset output from function load_data()
    	        
    output:
		  clean and return dataset 'df'
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    #For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
    for column in categories:
    	#set each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    	
    	# convert column from string to numeric
    	categories[column] = categories[column].astype(int)
    	#categories[column] = categories[column].astype(np.int)
    	
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()

    # return the cleaned dataframe 
    return df


def save_data(df, database_filename):

    """ 
    Write records stored in the DataFrame to a SQL database

    input:
    	df: cleaned dataset after function clean_data()
      database_filename: database name
    	         
    output:
      Saves the dataframe into sqlite database. Tablename is 'disaster_messages'
      Tablename can be changed in the code below
      ToDo: add Tablename as Input
    """

    try:
        engine = create_engine('sqlite:///' + database_filename)
        df.to_sql('DisasterResponseTable', engine, index=False)
    except Exception as e:
        print(e)

def main():

    if len(sys.argv) == 4:

        # parse the command line input
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading the data...')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning the data...')
        df = clean_data(df)
        
        print('Saving the data...')
        save_data(df, database_filepath)
        
        print('Cleaned input data saved to the database.')
    
    else:
        print('Example: '\
                'python process_data.py '\
                'messages.csv '\
                'categories.csv '\
                'DisasterResponse.db')


if __name__ == '__main__':
    main()