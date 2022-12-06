from nltk.tokenize import word_tokenize
from typing import final
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import math
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
query_index={}
cosine_sim=[]
def ProcessQuery(query_tokens):
    print("Processing Query")
    connecing_words = []
    query_words = []
    filtered_query=[]
    for w in total_words:
        query_index[w]=0
    for w in query_tokens:
        if w not in stopList:
            filtered_query.append(w)

    for w in filtered_query:
            lemma_query = lemma.lemmatize(w)
            i = filtered_query.index(w)
            filtered_query[i] = lemma_query

    for w in filtered_query:
        query_index[w]=1

    f=open("query_index.txt","w")
    f.write(str(query_index))
    f.close()

def calculate_cosine_sim(alpha):
    vector_x = 0.0
    vector_a = 0.0
    vector_b = 0.0
    cosine_score = []
    for i in range(1,449):
        for word in query_index:
            vector_x += (query_index[word] * postings[word]['tf-idf'][i]) 
            vector_a += query_index[word]**2
            vector_b += postings[word]['tf-idf'][i]**2
        cosine_sim.append(vector_x / (math.sqrt(vector_a) * math.sqrt(vector_b) ) )
        vector_x = 0.0
        vector_a = 0.0
        vector_b = 0.0
    doc_list1 = []
    score=1
    for i in cosine_sim:
        if i > alpha:
            cosine_score.append(i)
            doc_list1.append(score)
        score = score + 1
    
    return cosine_score, doc_list1
    


f = open("D:/UNIVERSITY/SEMESTER 6/IR/A2/Stopword-List.txt", "r")
stopList = f.read()
docList = 0
postings = {}
positional_index = {}
total_words=[]
# ps = PorterStemmer()
lemma = WordNetLemmatizer()
count = 0
for i in range(1, 449):
    f = open(f"D:/UNIVERSITY/SEMESTER 6/IR/A2/Abstracts/{i}.txt", "r")
    docList = f.read()
    docList = docList.replace("\n", " ")
    docList = docList.replace('-', " ")
    docList = docList.replace("/", " ")
    tokens = nltk.word_tokenize(docList)
    tokens = [tokens.lower() for tokens in tokens if tokens.isalnum()]
    filtered_sentence = [w for w in tokens if not w.lower() in stopList]
    filtered_sentence = []
    for w in tokens:
        if w not in stopList:
            filtered_sentence.append(w)
  
    for w in filtered_sentence:
        lemma_word=lemma.lemmatize(w)
        if lemma_word not in total_words:
            total_words.append(lemma_word)
        if lemma_word not in postings:
            postings[lemma_word]={
                        'tf' : [0]*449,  
                        'df' : 0,
                        'idf':0,
                        'tf-idf':[0]*449
                    }
            postings[lemma_word]['tf'][i] = 1
            postings[lemma_word]['df'] = 1
        else:
            if postings[lemma_word]['tf'][i] == 0 :  
                postings[lemma_word]['df'] = postings[lemma_word]['df'] + 1
                postings[lemma_word]['tf'][i] = 1
            else :
                postings[lemma_word]['tf'][i] = postings[lemma_word]['tf'][i] +1 
    
    for pos, w in enumerate(tokens):
        if w not in stopList:
            lemma_word = lemma.lemmatize(w)
            if lemma_word not in positional_index:
                positional_index[lemma_word] = []
                positional_index[lemma_word].append(1)
                positional_index[lemma_word].append({})
                positional_index[lemma_word][1][i] = [pos]
            else:
                positional_index[lemma_word][0] = positional_index[lemma_word][0]+1
                if i in positional_index[lemma_word][1]:
                    positional_index[lemma_word][1][i].append(pos)
                else:
                    positional_index[lemma_word][1][i] = [pos]

for doc_num in range(1,449):
        for word in postings:
            postings[word]['idf'] = math.log(449/(postings[word]['df']) , 10 )    
            postings[word]['tf-idf'][doc_num] = postings[word]['tf'][doc_num] * postings[word]['idf']


f = open("inverted_index.txt", "w")
f.write(str(postings))
f.close()

f = open("positional_index.txt", "w")
f.write(str(positional_index))
f.close()



def takeInput(query):
    query_tokens=nltk.word_tokenize(query)
    query_tokens=[query_tokens.lower() for query_tokens in query_tokens if query_tokens.isalnum()]
    ProcessQuery(query_tokens)
    cosine_sim_list,doc_list=calculate_cosine_sim(0.001)
    print("doc",doc_list)
    print("final",cosine_sim_list)
    return cosine_sim_list,doc_list


    
