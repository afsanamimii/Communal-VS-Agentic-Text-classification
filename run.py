from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd



app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == "POST":
        model = pickle.load(open(r"D:\DATA SCIENCE PROJECTS\Agentic vs communal\model.pkl", "rb"))
        
        with open ('dictionary.txt', 'rb') as file:
            
            Outlist = pickle.load(file)

        input_value=request.form['para']

        #print(input_value)



        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')


        
        corpus=[]
        review = re.sub('[^a-zA-Z]', ' ', input_value)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

        print(corpus)


        from sklearn.feature_extraction.text import CountVectorizer

        cvFile="D:\\DATA SCIENCE PROJECTS\\Agentic vs communal\\bow_train.pkl"
        cv = pickle.load(open(cvFile, "rb"))
        X_fresh=cv.transform(corpus).toarray()
        prediction=model.predict(X_fresh)
        print(prediction)

        class_probabilities = model.predict_proba(X_fresh)*100
        


        print("probability")
        print(class_probabilities)
        

        confidence_score=(class_probabilities)

        ######CONVERTING THE TEST INPUT INTO STRINGS################

        def convert(lst):
            
            return (lst[0].split())
 

        lst =  corpus
        test_dictionary=convert(lst)
        #print(format(class_probabilities[0],'f'))


       #########################################



       #################loading the dictionary valur(agentic/communal)###########################

        if(prediction[0]==1):
            with open ('dictionary_agentic.txt', 'rb') as file:
                print("openning agentic du=ictionary")
                train_dictionary = pickle.load(file)
                
                
        elif(prediction[0]==2):
            
            with open ('dictionary_communal.txt', 'rb') as file:
                print("openning communal du=ictionary")
                train_dictionary = pickle.load(file)

        ###########################################################################


        ##################Checking the common value ########################

        import numpy as np
        list1_as_set = set(train_dictionary)
        intersection = list1_as_set. intersection(test_dictionary)
        intersection_as_list = list(intersection)
        print("Highlight",intersection_as_list)

        
        #########################################################################

    else:
        prediction= None
        confidence_score=None
        intersection_as_list=None


    
    return render_template('home.html', prediction_text=prediction, conf=confidence_score,highlight=intersection_as_list)



if __name__ == "__main__":
    app.run(debug=True)
