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

The dataset has below three columns.
```
['id' 'sentiment' 'review']
```

- **id**: unique identifier; we don't need this field. 
- **sentiment**: has binary values (1 and 0). 1 for positive and 0 for negative. This is the label of the model. 
- **review**: Detailed review. This is the text or feature on which machine learning models will get trainned. 



