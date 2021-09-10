#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Note: 
#PDF encoding kept causing extreme data loss post conversion to txt, 
#hence the data was converted to txt outside of this python program
#This version has been included in the git repository for ease of access alongside its .pdf counterpart

import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx   
#from nltk.stem import PorterStemmer 

#Recieves the location of a research paper as a .txt file 
def read_paper(file):
  
    #text=convert_pdf_to_txt(file)

    #filedata = open("convertedFile.txt", "a")
    #filedata.writelines(text)

    #filedata.close()

    filedata = open(file, "r", encoding="utf8")
   
    #Removing
    paper_without_coverpage = filedata.read().replace('\n', ' ').split("ABSTRACT")
    print("Title Page: \n" + paper_without_coverpage[0])

    #Removing Acknowledgements and bibliography
    paper_short = paper_without_coverpage[1].split("ACKNOWLEDGMENTS") 
    
    #Separating sentences i.e. paper is a list of sentences
    paper = paper_short[0].split(". ")
    sentences = []

    for sentence in paper:
        
        #remove whitespace
        #sentence=sentence.strip()
            
        #remove header
        sentence=sentence.replace("Text Summarization Techniques: A Brief Survey arXiv, July 2017, USA"," ")
        
        #remove non-ascii and digits
        sentence=sentence.replace("(\\W|\\d)"," ")
        
        #[^a-zA-Z] matches to all strings that contain a symbol that is not a letter
        sentences.append(sentence.replace("[^a-zA-Z]", "").split(" "))     
        
        # remove markup
        sentence=sentence.replace("[.*?]","")
            
        #print('\n'+sentence)

#Note: Stemmer commented out since stemming an entire article is a lot of manual work 
#and I feel there must be a more convenient way to approach this - but that is outside of my knowledge scope at this time
#This is also why Lemmatization was not attempted
    
        #porter_stemmer=PorterStemmer()    
        
        #words=["summaries", "summary", "summarize", "summarization", "summerizers"]
        #stemmed_words=[porter_stemmer.stem(word=word) for word in words]
        #stemdf= pd.DataFrame({'original_word': words,'stemmed_word': stemmed_words})
        
    sentences.pop() 
    return sentences

#Using cosine distance instead of euclidian since it accounts for similarity without being affected by word repetition
def find_sentence_similarity_with_cosine_similarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    #lowercasing
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]

 
    all_words = list(set(sentence1 + sentence2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sentence1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sentence2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
#The similarity matrix will help account for the sentences that are too similar to each other since there may be different sentences expressing the same message
def create_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = find_sentence_similarity_with_cosine_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Using textrank under assumption only one article will be considered and not multiple articles - in which case Lex ranking could be usedbv 
def create_summary(file, n): #n signifies the number of first best ranked sentences to be used to create a summary
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_paper(file)
    #Generates a similarity matrix accross sentences
    sentence_similarity_martix = create_similarity_matrix(sentences, stop_words)

    #Ranks sentences in the similarity martix with a pre built pagerank algorithm
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph, max_iter=2000) #Max iteration increased since it is only 100 by default

    #Sorting the resulting ranks and selecting the top n sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    
    #print("Indexes of top ranked sentence order:  ", ranked_sentence)    
        
    for i in range(n):
      summarize_text.append(" ".join(ranked_sentence[i][1])) #joining the sentences into a single body of text

    print("\n--------------------------------------------------------------------------------------------------------")
    print("\nSummary: \n", ". ".join(summarize_text))

# test
#12 is 5% of the total sentences in this paper - after removing the acknowledgements and bibliography hence it is a fairly low compression ratio
create_summary( "C:/Users/Administrator/Downloads/TextSummarization.txt", 12) 

#The precision ratio is hard to estimate, but from reading the generated summary, there is a lot of room for improvement. This remains the case even with higher compression ratios i.e. 10%
#create_summary( "C:/Users/Administrator/Downloads/TextSummarization.txt", 24) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




