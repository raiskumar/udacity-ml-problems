## Naive Bayes Classifier
One particular feature of Naive Bayes is that it’s a good algorithm for working with text classification. When dealing with text, it’s very common to treat each unique word as a feature, and since the typical review vocabulary is many thousands of words, this makes for a large number of features. The relative simplicity of the algorithm and the independent features assumption of Naive Bayes make it a strong performer for classifying texts.

The Naive Bayes classifier uses the Bayes Theorem to select the outcome with the highest probability. This classifier assumes 
the features(in this case the words) are independent and hence the word naive.

The Naive Bayes classifier for this problem says that the probability of the label (positive or negative) for the given review
text is equal to the probability of the text given the label, times the probability a label occurs, everything divided by 
the probability that this text is found. 

![naive_bayes_1.png](naive_bayes_1.png)

Text in our case is collection of words. So above equation can be expressed as:

![naive_bayes_2.png](naive_bayes_2.png)

We want to compare the probabilities of the labels and choose the one with higher probability. The denominator, i.e. the term P(word1, word2, word3…) is equal for everything, so we can ignore it. Also, as discussed above **there is no dependence between words in the text** (not possible always as few words mostly appear together but we can ignore such abberations); so equation can be re-written as:

![naive_bayes_3.png](naive_bayes_3.png)

* **P(label=positive)** is the fraction of the training set that is a positive text;
* **P(word1|label=negative)** is the number of times the word1 appears in a negative text divided by the number of times the word1 appears in every text.
