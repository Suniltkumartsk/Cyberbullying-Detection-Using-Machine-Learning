# Cyberbullying Detection Using Machine Learning

This project is a simple machine learning system for detecting cyberbullying in text using Python and scikit-learn. It uses a logistic regression model trained on a labeled dataset of tweets.

## Features

- Cleans and preprocesses text data.
- Uses TF-IDF vectorization with n-grams.
- Trains a logistic regression classifier.
- Evaluates model performance (accuracy, classification report, confusion matrix).
- Provides an interactive prompt for users to enter sentences and get predictions.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

Install dependencies with:
```
pip install pandas numpy scikit-learn
```

## Dataset

The code expects a CSV file named `cyber.csv` in the same directory, with at least the following columns:
- `tweet_text`: The text to analyze.
- `cyberbullying_type`: The label/class for each text.

## How to Run

1. Place `cyber.csv` in the project directory.
2. Run the script:
   ```
   python cyberml.py
   ```
3. After training, enter sentences at the prompt to get predictions. Type `exit` to quit.

## Notes

- The model's accuracy depends on the quality and size of your dataset.
- You can improve detection by expanding the dataset and enhancing text preprocessing.

## Example

```
Enter a sentence: you are so stupid
Predicted Class: bullying

Enter a sentence: have a nice day
Predicted Class: not_bullying

Enter a sentence: exit
Exiting program...
```# Cyberbullying-Detection-Using-Machine-Learning
