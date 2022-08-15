from naive_bayes import NaiveBayesNameClassifier
from lstm import LSTMNameClassifier

""" Test the already trained model on your custom data """
# model was trained 80% of name_gender.csv
loaded_nbayes = NaiveBayesNameClassifier(load_model='nbayes_name_gender_split80')
#loaded_nbayes.test_model(train_data='my_custom_data.csv')

# try out a name
loaded_nbayes.test_nbayes_name(name='Bernhard')


""" Train and test the model from scratch """
# model is trained 80% of name_gender.csv
trained_nbayes = NaiveBayesNameClassifier(train_data='name_gender.csv', train_split=.8)
# test is on the last 20% of name_gender.csv
test_accuracy = trained_nbayes.test_nbayes_data()
print('Test accuracy of trained_nbayes:', test_accuracy)



trained_lstm = LSTMNameClassifier(model_to_load=None, train_data='name_gender.csv', train_split=.8)
