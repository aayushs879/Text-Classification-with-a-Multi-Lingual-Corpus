# Soc-Gen-Complaint-Status-Tracking
Societe Generale (SocGen) is a French multinational banking and financial services company. With over 1,54,000 employees, based in 76 countries, they handle over 32 million clients throughout the world on a daily basis.  They provide services like retail banking, corporate and investment banking, asset management, portfolio management, insurance and other financial services.  While handling customer complaints, it is hard to track the status of the complaint. To automate this process, SocGen wants you to build a model that can automatically predict the complaint status (how the complaint was resolved) based on the complaint submitted by the consumer and other related meta-data.
Data Description
The dataset consists of three files: train.csv, test.csv and sample_submission.csv.

Complaint-ID- Complaint Id

Date received - Date on which the complaint was received

Transaction-Type - Type of transaction involved

Complaint-reason - Reason of the complaint

Consumer-complaint-summary - Complaint filed by the consumer - Present in three languages :  English, Spanish, French

Company-response - Public response provided by the company (if any)

Date-sent-to-company - Date on which the complaint was sent to the respective department

Complaint-Status - Status of the complaint (Target Variable)

Consumer-disputes - If the consumer raised any disputes



Approach - 

First of all, obtained importance of different features using sklearn's ExtraTreesClassifier. The numbers were not satisfying.
Then started with Complaint summary - firstly detected all languages using python's langdetect package.
Cleaned the text using standard nlp techniques i.e expanding contractions, punctuation removal, Lemmatization etc. For stopwords removal made a set of stopwords of all the used languages i.e English, Spanish and French. 

Created a separate dictionary for each language with the key as index and value as text content.
Then used facebook's fastText trained vectors for generating word embeddings for French and Spanish and GloVe for English.
The final word embedding of a sentence was a Tf-Idf weighted average of most occuring words where no. of words to be selected for each corpus was proportional to its distribution in dataset.
Finally after getting the word vectors Trained a separate Deep Neural Network for each language using Tensorflow with the architecture
(input - 2000 - 2000 - 2000 - 1000 - n_class )
The loss function used is weighted cross entropy as there was quite an imbalance in the class distribution.
Finally it is optimized using rmsprop optimizer and the weighted recall score obtained on hackerearth on the submission script was 0.81.
