# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
import numpy as np
import re

"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca18pal" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features", "bpe"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

# import data using pandas
# return as numpy array
# shape with 3 columns for ID, review, and rating
def import_data(file_path):
    dataframe = pd.read_csv(file_path, sep='\t')
    return dataframe.to_numpy()

# converts all reviews to lower case
def to_lower_case_sentances(data):
    for entry in data:
        entry[1] = entry[1].lower()
    
    return data

# tokenises every review
# removes all punctuation except for "!"
def tokenise_sentances(data):
    for entry in data:
        entry[1] = re.findall(r"[\w']+|[!]", entry[1])

    return data

# puts preprocessing steps together
def process_data(data):
    data = to_lower_case_sentances(data)
    data = tokenise_sentances(data)
    
    return data

# counts various frequencies within the data
# word_frequency_per_class: dictionary with word (feature) as key, 
# each containing how many times the word appears in each class
# words_per_class: how many words in each class
# class_frequency: how many instances of each class
# line 75 initialises counts as 1s and not 0s in order to apply smoothing
def count_frequencies(data, number_classes):
    word_frequency_per_class = {}
    words_per_class = np.zeros(number_classes)
    class_frequency = np.zeros(number_classes)

    for entry in data:
        class_frequency[entry[2]] += 1
        words_per_class[entry[2]] += len(entry[1])
        
        for word in entry[1]:
            word_frequency_per_entry = np.zeros(number_classes)
            
            word_frequency_per_entry[entry[2]] += entry[1].count(word)
            
            if word not in word_frequency_per_class:
                word_frequency_per_class[word] = np.add(word_frequency_per_entry, np.ones(number_classes))
            else:
                word_frequency_per_class[word] = np.add(word_frequency_per_class[word], word_frequency_per_entry)
                
    return (word_frequency_per_class, words_per_class, class_frequency)
          
# calculates the prior frequencies for each class
# done by entries in each class / total entries
def calculate_prior_probabilities(class_frequency, length_of_data, number_classes):
    prior_probabilities = np.zeros(number_classes)
    
    for i in range(number_classes):
        prior_probabilities[i] = class_frequency[i]/length_of_data
        
    return prior_probabilities

# calculates likelihood for each word to be within each class
# done by taking the number of times that a word appears within the class, 
# divided by the words in that class, plus the total of feature words (for smoothing)
def calculate_likelihood_per_feature(word_frequency_per_class, words_per_class, number_classes):
    word_likelihood_per_class = {}
    
    for word in word_frequency_per_class:
        word_likelihood_per_class[word] = np.divide(word_frequency_per_class[word], words_per_class + len(word_frequency_per_class))
    
    return word_likelihood_per_class
        
def calculate_prosterior_probabilities(prior_probabilities, likelihoods, data, number_classes):
    id_prosterior_probabilities = {}
    
    for entry in data:
        id_prosterior_probabilities[entry[0]] = prior_probabilities
        
        for word in entry[1]:
            if(word in likelihoods):
                id_prosterior_probabilities[entry[0]] = id_prosterior_probabilities[entry[0]] * likelihoods[word]
            
    return id_prosterior_probabilities

def classify(id_prosterior_probabilities):
    classified_ids = {}
    
    for entry in id_prosterior_probabilities:
        classified_ids[entry] = np.argmax(id_prosterior_probabilities[entry])
        
    return classified_ids

def calculate_confusion_matrix(classified_ids, data, number_classes):
    confusion_matrix = np.zeros((5,5))
    
    for entry in data:
        confusion_matrix[entry[2]][classified_ids[entry[0]]] += 1
        
    return confusion_matrix

def calculate_f1(confusion_matrix, number_classes):
    f1_scores = np.zeros(number_classes)
    
    precision = np.zeros(number_classes)
    recall = np.zeros(number_classes)
    
    for i in range(number_classes):
        precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
        recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[:,i])
        
        f1_scores[i] = (2*precision[i]*recall[i])/(precision[i] + recall[i])
        
    return np.mean(f1_scores)
    
def calculate_accuracy(confusion_matrix, number_classes):
    tp_and_tn = 0
    
    for i in range(number_classes):
        tp_and_tn += confusion_matrix[i][i]
        
    return tp_and_tn/np.sum(confusion_matrix)

                                       
def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features; "bpe" to use BPEs or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    
    training_data = import_data(training)
    development_data = import_data(dev)
    test_data = import_data(test)

    training_data = process_data(training_data)
    development_data = process_data(development_data)
    test_data = process_data(test_data)
    
    frequencies = count_frequencies(training_data, number_classes)
    
    print(frequencies[0]["good"])
    
    prior_probabilities = calculate_prior_probabilities(frequencies[2], len(training_data), number_classes)
    likelihoods = calculate_likelihood_per_feature(frequencies[0], frequencies[1], number_classes)  
    
    dev_id_prosterior_probabilities = calculate_prosterior_probabilities(prior_probabilities, likelihoods, development_data, number_classes)
    test_id_prosterior_probabilities = calculate_prosterior_probabilities(prior_probabilities, likelihoods, test_data, number_classes)
    
    dev_classified_ids = classify(dev_id_prosterior_probabilities)
    test_classified_ids = classify(test_id_prosterior_probabilities)
    
    confusion_matrix = calculate_confusion_matrix(dev_classified_ids, development_data, number_classes)
    
    f1_score = calculate_f1(confusion_matrix, number_classes)
    accuracy = calculate_accuracy(confusion_matrix, number_classes)
    
    
    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f\t%f" % (USER_ID, number_classes, features, f1_score, accuracy))

if __name__ == "__main__":
    main()