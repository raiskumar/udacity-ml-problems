# Machine Learning Engineer Nanodegree
## Capstone Proposal
Siddheshwar Kumar  
April 12, 2018

## Proposal


### Domain Background

Around 2 years back, I participated in my company Hackathon where I worked on the **sentiment** and **intent** analysis for a chatbot. I had used Stanford NLP library but during the contest, I could only scratch the surface of NLP. And then again, around an year back I explored **sentiment analysis** on the tweets; but again couldn't make much progress. In either cases I couldn't understand the field much, but it gave me motivation to explore the field further and that's one of the prime reason for me to do this course.

**Sentiment Analysis** is one of the most interesting domain of machine learning. I am planning to use this oportunity to explore this field. One common use of sentiment analysis is to figure out if a text expresses negative or positive feelings. 

### Problem Statement

**Perform sentiment analysis using machine learning techniques.** On internet a lot of content is available which gives review or feedback of movies. Now, for a person to decide whether to go for a movie or not is not so trivial if he/she has to go through 
countless tweets, facebook posts/comments and IMDB reviews. 

Machine learning can play a big role in churning out all these data from diverse sources and just providing an overall positive or negative feedback of the movies i.e. *in short weather it's thumbs up or down*. Machine learning can also help to classify a movie review in more than two categories like average, good, very good, bad etc. As a user it will be really helpful to just know a very objective feedback based on the subjective reviews provided by other users.

**For this capstore project, I will be using IMDB dataset.**

### Datasets and Inputs

I will be using the data from an old Kaggle competition **Bag of Words Meets Bags of Popcorn** (https://www.kaggle.com/c/word2vec-nlp-tutorial). This dataset contains 25,000 labeled training reviews, 50,000 unlabeled training reviews, and 25,000 testing reviews. 

The dataset has below three columns/fields.
```
['id' 'sentiment' 'review']
```

- **id**: Unique identifier for each entry in the dataset; we don't need this field for modelling. 
- **sentiment**: Contains binary values (1 and 0). 1 for positive and 0 for negative. This is the label of the model. 
- **review**: Detailed review for movies. This is the text or feature on which machine learning models will get trainned. 

**Unlabeled training** data and **testing data** doesn't have _sentiment_ field. So for this project, I will rely only on **labelled training** data for training model as well as testing the model. The file (*labeledTrainData.tsv*) is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  

I am planning to use 10% of data testing and will report accuracy of the model on the performance of this data.

#### Input dataset: labeledTrainData.tsv
- **Dataset size for Training the model:** 90% of 25,000 = 22,500
- **Dataset size for Testing the model:** 10% of 25,000 = 2,500


### Solution Statement

I plan to use Deep Learning techinques as the final solution. Number of words in the reviews field is not fixed and also the meaning doesn't depend directly on words but also on the context. So the same word can have different meaning depending on the surrounding words. Recurrent Neural Network is a type of Deep learning techqniue which deals with time series data of variable length and have had good success off late in area of natural languages. As part of this project, I want to learn about RNN and apply to this problem. But, at the same time I would also like to apply normal machine learning techniques to understand how they peform in this given problem. 


### Benchmark Model

I plan to compare the performance of RNN (or LSTM: Long-Short Team Memory) with that of traditional approaches like Naive Bayes. I will compare the accuracy or mean-squared error of each technique to analyse which one is more effective and try to find out the reasom for the same. Also, this is a Kaggle problem and the leader board shows good success rate. So, planing to taget accuracy of at least 90%. 


