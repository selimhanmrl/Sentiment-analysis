# Sentiment-analysis
My First NLP project


For Dataset : https://www.kaggle.com/kazanova/sentiment140


		



      
In this project firstly I used Google Colab. I write it on Jupiter notebook but because of some reason about ram using I tried Google Colab and work on there. 

As you see in my code I uploaded csv file to my google drive and I read with pandas library.
For checking I look their column name and there was not named before and I gave name but there was unnecessary informations like user name etc. I need only sentiment and Tweet texts.
Because of decreasing execution time I took part of that data frame. It was half of positive tweet other half part is negative tweets. After I showed word clouds for seeing differences of words positive and negative tweets has.

I started to clean tweets. Firstly I make all tweets lower case. Python has lowercase precision so it can effect performance. And after checking some tweets I saw there are some URLs so I cleaned all URLs which has “www. Or https” also after URL it has nicknames there could be a problem it I clean only ‘@‘ symbol. Because it can be ‘@iamsohappy’ and after cleaning ‘@‘ symbol it will be “iamhappy’ and it can be problem. And I cleaned words which has ‘@‘ at the beginning.after I continued with symbols like “. , \ ! …”. It does not mean much emotions. Just emojis can make a positive or negative tweets but I did not work on it. After I cleaned numbers. With nltk library it has own list of ‘English stop words’ and I cleaned them in tweets. It is most common letters and because of grammar it uses but it does not share feeling so it effect so little tweets positive or negative. After cleaned stopwords I checked again which words are using much and I notice that ‘im’ and ‘u’ words using in social media much and it is “I’m and you’ they was not in stop words list and I added them and cleaned after them. I tokenize letters After I stemming word for showing roots of words and analize better and make them together again.

All processes above are for make text understandable and work easily on texts. It named as preprocessing.

After that I started to process of learning and and test algorithms. Before process I vectorized data frame and I used CountVectorizer for this process. Because count vectorize make matris for probability of using words. I had some problems these were not error but when I see vectorizer features names there is empty values and many meaningless words etc.

End of that code I tried algorithm and measure their accuracies. Accuracy is not enough for comparing but we know our dateset and it is half of positive and half of negative and because of it accuracy can be good for comparing. Also because of dataset size accuracy changed. On logistic regression it decreased to 0.74 from 0.76 Before starting project I read about Logistic Regression is best algorithm or sentiment analysis. 

I tested on:

	- Logistic Regression   	with 0.7471 accuracy

	- Naive Bayes	     	      with 0.7406 accuracy

	- Random Forests	      	with has 0.7285 accuracy

 	- XGBoost algorithms	    with has 0.6808 accuracy

As you see on my test there is no big differences but Logistic regression is better algorithm. Also for process time it has less time. And if I write orders time as longer to shorter.
Random Forests - XGBosst - Logistic Regression - Naive Bayes. 


Selimhan Meral


