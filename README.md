Ham or Spam?
Introduction to NLP with Python
In this article, we will be looking at one of the basics of Natural Language Processing, which is to train a classifier that is able to differentiate one class from another, in this case Spam or not-Spam. We will have to extract features from text, and select a classification algorithm that works best for us. We will be using Python 3 and the dataset is the SMS Spam Collection which tags 5,574 text messages based on whether they are “spam” or “ham” (not spam).

Our goal is to build a predictive model which will determine whether a text message is spam or ham.

Mailbox

![](https://camo.githubusercontent.com/8314123967957ef424a6e288dc10aec8269ea29f/68747470733a2f2f63646e2e706978616261792e636f6d2f70686f746f2f323031352f31312f31372f32332f33332f6d61696c2d313034383435325f5f3334302e6a7067)

What is Spam?
Spam is, according to wikipedia, it's described as “the use of electronic messaging systems to send unsolicited bulk messages, especially advertising, indiscriminately.” The word was coined sometime in 2001 or 2002, by the guys working on SpamBayes, the Python probabilistic classifier. Although there are more formal definitions, the key word is “unsolicited”. This means that you did not ask for messages from this source. So if you didn’t ask for the mail it must be spam, Right? Maybe, but if we are looking to differentiate one class from another, we need to start finding patterns.

The data and features
Taking a first look at the data will give us valuable insights and if we do it in a systematic way it is called EDA, Exploratory Data Analysis. When building a classificator the most important value to look at is how balanced is our data. This is because it gives us a sense on how much our classificator is helping us. For example, if we have a 90% of class A and 10% of B, we may have a classificator with 90% accuracy that hasn’t learned anything besides predicting A every time it sees something new.

This particular data set also has 4821 messages labelled as “ham” and 746 messages labelled as “spam”, which is the 87% and 13% repectlively. The class imbalance will become important later when assessing the strength of our classifier.

Another thing we are looking at is the length of the message and the ratio of punctuation to letters. The length of the message will give us an idea of how many characters we will be dealing in every message, and the ratio of punctuation are features that we will be using to predict whether they are spam or not.

We will be looking to extract every feature that allows us to classify better, and remove everything that makes us do worse. This is the idea that we will apply when removing stop words and stemming words.

This is sometimes what we are looking for when we perform normalization of the data. When used correctly, it reduces noise, groups terms with similar semantic meanings and reduces computational costs by giving us a smaller matrix to work with. This matrix will be obtained using a vectorizer. Let’s dive into a more detailed explanation of each method.

Stopwords
When going through text, there are words that are used all the time like connectors, articles and so on. In natural language processing, stop words are words which are filtered out before or after processing of data. These are some of the most common, short function words, such as the, is, at, which, and on. The basic idea is that if they are present in most of the texts, they are not adding information that allow us to see what makes every class different.

Though "stop words" usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list. In our case we will be using the list of words that NLTK library provides.

Any group of words can be chosen as the stop words for a given purpose, and we can also add custom words to our list.

Stemming
Stemming is used to reduce every word into its root to group words that mean the same thing. The idea of stemming is a sort of normalizing method. We will use it to group words that have the same root but are expressed in a different tense. Many variations of words carry the same meaning, other than when tense is involved.

The reason why we stem is to shorten the lookup, and normalize sentences.

Consider the phrase “I was taking a ride in the car” and “I was riding in the car”. This sentence means the same thing. The difference lies in the tense or termination of the verb “ride”. The “ing” at the end of “riding” denotes a clear past-tense, so is it truly necessary to differentiate between ride and riding, in the case of just trying to figure out the meaning of what this past-tense activity was? No, not really.

This might be just one minor example, but imagine every word in the English, every possible tense and affix you can put on a word. Having individual dictionary entries per version would be highly redundant and inefficient, especially since, once we convert to numbers, the "value" is going to be identical.

For our example, we will be using the Porter stemmer implemented in the NLTK library, one of the most popular stemming algorithms, which has been around since 1979.

TF-IDF
Term Frequency - Inverse Document Frequency (TF-IDF) is a statistical measure that tries to evaluate how relevant a word is to a document in a collection of documents. It has many uses, most importantly in Natural Language Processing, where used to scoring words in machine learning algorithms.

The relevance of each word is analyzed by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.

tf-idf

Term frequency is the number of times a word appears in a document (in this case document meaning each message) divided by the total number of words in the document. Every document has its own term frequency. The inverse document frequency is the second term which evaluates how frequent is this term in the rest of the documents.

TF-IDF was invented for document search and information retrieval and it works by assigning a value to each word and increasing it proportionally to the number of times a word appears in a document, but is offset by the number of documents that contain the word. So, words that are common in every document rank low even though they may appear many times, since they don’t mean much to that document in particular.

However, if a certain word appears many times in a document, while not appearing many times in others, it probably means that it’s very relevant.

Vectorizing the Text with TF-IDF
We need to transform the text into something that the algorithm can work with, and those are numeric vectors. The process of turning text into numbers is commonly known as vectorization or embedding. Vectorizers are functions which map words onto vectors of real numbers. The vectors form a vector space where all the rules of vector addition and measures of similarities apply. We will use a vectorizer which transforms a text into a vector representation given a certain method, in this case TF-IDF.

While most vectorizers have their unique advantages, it is not always clear which one to use. In this case, the TF-IDF vectorizer was chosen for its simplicity and efficiency in vectorizing documents such as text messages.

Building a classifier and choosing and algorithm
The next step is to select the type of classification algorithm to use. We will choose two candidate classifiers and evaluate them against the testing set to see which one works the best. For this we have selected two algorithms which are Random Forest and Gradient Boosting, both implemented in the Scikit-learn library.

Random Forest
Let’s understand the algorithm in layman’s terms. Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy to use algorithm. A forest consists of trees. It is said that the more trees it has, the more robust a forest is. The Random forests algorithm creates decision trees on randomly selected data samples, gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good indicator of the feature importance.

It technically is an ensemble method (based on the divide-and-conquer approach) of decision trees generated on a randomly split dataset. This collection of decision tree classifiers is also known as the forest. The individual decision trees are generated using an attribute selection indicator such as information gain, gain ratio, and Gini index for each attribute. Each tree depends on an independent random sample. In a classification problem, each tree votes and the most popular class is chosen as the final result. In the case of regression, the average of all the tree outputs is considered as the final result. It is simpler and more powerful compared to the other non-linear classification algorithms.

The algorithm works in four steps:

Select random samples from a given dataset.
Construct a decision tree for each sample and get a prediction result from - each decision tree.
Perform a vote for each predicted result.
Select the prediction result with the most votes as the final prediction.
It has advantages like:

Is considered to be highly accurate and robust because of the number of - decision trees participating in the process.
Reduces the overfitting problem because it takes the average of all the predictions, which cancels out the biases.
We can get the relative feature importance, which helps in selecting the most contributing features for the classifier.
But also has disadvantages:

Is slow in generating predictions because it has multiple decision trees. Whenever it makes a prediction, all the trees in the forest have to make a prediction for the same given input and then perform voting on it. This whole process is time-consuming.
The model is difficult to interpret compared to a decision tree, where you can easily make a decision by following the path in the tree.
Gradient Boosting
Boosting is a general ensemble technique that involves sequentially adding models to the ensemble where subsequent models correct the performance of prior models. Gradient boosting classifiers are machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting.

Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

Gradient boosting is often the main, or one of the main, algorithms used to win machine learning competitions like Kaggle on tabular and similar structured datasets because of its efficiency.

Final evaluation and conclusion
We will use three different metrics to evaluate the performance of our classifiers. This is because we are dealing with a highly imbalanced dataset, and calculating just precision would be misleading for us. Therefore we included recall and accuracy, which give us a sense on how much of the least found category we are able to detect.

After training both of the classifiers we obtain the next metrics:

Random Forest:

Precision: 1.0
Recall: 0.81
Accuracy: 0.975
Gradient Boosting:

Precision: 0.889
Recall: 0.816
Accuracy: 0.962
We can observe that Random Forest prevailed at two of the three metrics, with impeccable precision. Therefore, if we can deal with all the possible disadvantages, is the algorithm that we would consider to be the best.
