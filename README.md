# Sentiment Analysis

## Objective
The main objective of this project is to determine the attitude of the people is positive, negative or neutral towards the subject of interest and represent it in the form of Pie-chart.

## Scope 
- This project can be used in social media monitoring, market research, customer feedback, brand monitoring, customer services and other fields which include social interaction of people.
- It would be helpful to the companies, political parties as well as to the common people.
- It would be helpful to political party for reviewing about the program that they are going to do or the program that they have performed.
- Similarly companies also can get review about their new product or newly released hardware or software.
- Also the movie maker can take review on the currently running movie by analyzing tweets which can be included in brand monitoring.

## Implementation
Implemented the model to provide a sentiment score between 0 to 1 with 0 being very negative and 1 being very positive. This was done by building a multi-class classification model i.e 10 class, one class for each decile.
There are 5 major steps involved in building this model:
1. Get data
2. Generate embedding's
3. Model architecture
4. Model parameters
5. Train and test the model 
6. Run the model

### 1. Get data
- Used Stanford sentiment treebank data.
- The dataset ‘dictionary.txt’ consists of 239,233 lines of sentences with an index for each line.
- Another file ‘sentiment_values.txt’ which contains sentiment score along with index number.
- The data is splited into 3 parts :
  - Train.csv : the main data which is used to train the model. (80% of overall data)
  - Val.csv : the validation dataset to be used to ensure the model does not overfit. (10%)
  - Test.csv : this is used to test the accuracy of the model post training. (10%)

### 2. Generate Embeddings
- Prior to train model we converted each of the words into a word embedding.
- One can think of word embedding as numerical representation of words to enable our model to learn.
- We used pre-trained word embedding model known as GloVe.
<br/>

![image](https://user-images.githubusercontent.com/37670032/149420293-6c3668cd-3304-44d2-82d3-756459f73746.png)

### 3. Model architecture
- To train the model we used RNN (Recurrent Neural Network), known as LSTM (Long Short Term Memory).
- The main advantage of this network is that it is able to remember the sequence of past data i.e. word in our case in order to make a decision on the sentiment of the word.
- We created network using keras.
- In order to estimate parameters such as dropout, no of cells etc we have performed a grid search with different parameter values and chose the parameters with best performance.
![image](https://user-images.githubusercontent.com/37670032/149420527-5155b5eb-941b-4481-9fc0-b00ab8b6eca6.png)

### 4. Model parameters
- Activation Function: we have used ReLU (rectified linear activation unit) as the activation function. ReLU helps complex relationships in the data to be captured by the model.
- Optimiser: We use adam optimiser, which is an adaptive learning rate optimiser.
- Loss function: We have trained a network to output a probability over the 10 classes using Cross-Entropy loss, also called Softmax Loss.

### 5. Train and test the model
- We have run the training on a batch size of 2000 items at a time.
- The training is set to run for 25 epochs. 
- We have enabled early stopping to overcome overfitting.
- Early stopping is a method that allows us to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out/validation dataset.
- The model on the test set of 10 class sentiment classification provides a result of 78.6% accuracy.
