# Disaster Response Pipeline Project
Udacity Data Scientist Nanodegree Project

### Table of Contents

1. [Project Summary](#summary)
2. [File Structure](#file)
3. [Instructions](#instructions)
4. [Screenshots](#screenshots)

## Project Summary <a name="summary"></a>

Using a data set containing real messages that were sent during disaster events provided by <a href="www.figure-eight.com">Figure Eight </a>, the project creates a machine learning pipeline to categorize the events so that you can send the messages to an appropriate disaster relief agency.

The project output includes a web app where as an emergency worker can input a new message and get classification results in several categories. The web app displays various visualizations of the data.

## File Structure <a name="file"></a>

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model

- README.md
```

## Instructions: <a name="instructions"></a>

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots: <a name="screenshots"></a>

#### Flask App Output
![Flask App screenshot1](img/1.png)

#### Example output from a message inquiry
![Flask App screenshot2](img/2.png)
