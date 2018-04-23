## Naive Bayes Classifier
The Naive Bayes classifier uses the Bayes Theorem to select the outcome with the highest probability. This classifier assumes 
the features(in this case the words) are independent and hence the word naive.

The Naive Bayes classifier for this problem says that the probability of the label (positive or negative) for the given review
text is equal to the probability of the text given the label, times the probability a label occurs, everything divided by 
the probability that this text is found. 

![naive_bayes_1.png](naive_bayes_1.png)

Text in our case is collection of words. So above equation can be expressed as:

![naive_bayes_2.png](naive_bayes_2.png)

The denominator, i.e. the term P(word1, word2, word3…) is equal for everything, so we can ignore it. Also, as discussed above **there is no dependence between words in the text**; so equation can be re-written as:

![naive_bayes_3.png](naive_bayes_3.png)

* **P(label=positive)** is the fraction of the training set that is a positive text;
* **P(word1|label=negative)** is the number of times the word1 appears in a negative text divided by the number of times the word1 appears in every text.
