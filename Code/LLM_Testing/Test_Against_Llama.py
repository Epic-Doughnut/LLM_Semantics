from groq import Groq
import csv
import requests
import re
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import pandas as pd
import ast
import math


class Llama_Test:
    def __init__(self):
        # nltk.download('wordnet')
        # Stop words
        self.STOP_WORDS = set(
            """
        a about above across after afterwards again against all almost alone along
        already also although always am among amongst amount an and another any anyhow
        anyone anything anyway anywhere are around as at

        back be became because become becomes becoming been before beforehand behind
        being below beside besides between beyond both bottom but by

        call can cannot ca could

        did do does doing done down due during

        each eight either eleven else elsewhere empty enough even ever every
        everyone everything everywhere except

        few fifteen fifty first five for former formerly forty four from front full
        further

        get give go

        had has have he hence her here hereafter hereby herein hereupon hers herself
        him himself his how however hundred

        i if in indeed into is it its itself

        keep

        last latter latterly least less

        just

        made make many may me meanwhile might mine more moreover most mostly move much
        must my myself

        name namely neither never nevertheless next nine no nobody none noone nor not
        nothing now nowhere

        of off often on once one only onto or other others otherwise our ours ourselves
        out over own

        part per perhaps please put

        quite

        rather re really regarding

        same say see seem seemed seeming seems serious several she should show side
        since six sixty so some somehow someone something sometime sometimes somewhere
        still such

        take ten than that the their them themselves then thence there thereafter
        thereby therefore therein thereupon these they third this those though three
        through throughout thru thus to together too top toward towards twelve twenty
        two

        under until up unless upon us used using

        various very very via was we well were what whatever when whence whenever where
        whereafter whereas whereby wherein whereupon wherever whether which while
        whither who whoever whole whom whose why will with within without would

        yet you your yours yourself yourselves
        """.split()
        )

        self.contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
        self.STOP_WORDS.update(self.contractions)

        for apostrophe in ["‘", "’"]:
            for stopword in self.contractions:
                self.STOP_WORDS.add(stopword.replace("'", apostrophe))
        
        self.stops = list(list(set(self.STOP_WORDS)) + ["http"])



        # Setting up the emotion dictionary
        fileEmotion = r"C:\Users\natha\Programming\LLM_Research\LLM_Semantics\Code\emotion_itensity.txt"
        table = pd.read_csv(fileEmotion,  names=["word", "emotion", "itensity"], sep='\t')
        self.emotion_dic = dict()
        self.lmtzr = WordNetLemmatizer()
        for index, row in table.iterrows():
            #add first as it is given in the lexicon
            temp_key = row['word'] + '#' + row['emotion']
            self.emotion_dic[temp_key] = row['itensity']

            #add in the normal noun form
            temp_key_n = self.lmtzr.lemmatize(row['word']) + '#' + row['emotion']
            self.emotion_dic[temp_key_n] = row['itensity']
            
            #add in the normal verb form
            temp_key_v = self.lmtzr.lemmatize(row['word'], 'v') + '#' + row['emotion']
            self.emotion_dic[temp_key_v] = row['itensity'] 

    def Cleanup_Description(description):
        filtered = re.sub("[^a-zA-Z]+", " ", description) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        filtered = re.sub(pat, '', filtered).lower() #  convert to lowercase
        filtered = ' '.join(filtered.split())
        return filtered

    def Filter_Sample(file_path, lower_age, upper_age):
        """
        The purpose of this function is to take the data set data and vector data from "complete_pl_output.csv" and
        filter that down into the age group that we want to test with the LLM's. The function will output two files
        from the data: a Sample_Data.txt file with book descriptions that the LLM can read in and produce recommendations
        for, and a Filtered_Data.txt file which will have all the data for all the descriptions written to Sample_Data.txt.
        """
        with open(file_path, encoding="utf-8") as raw_data:
            with open("Sample_Data.txt", "w") as descriptions:
                with open("Filtered_Data.txt", "w") as filtered_raw:
                    csv_reader = csv.reader(raw_data)
                    writer_descriptions = csv.writer(descriptions)
                    writer_filtered_raw = csv.writer(filtered_raw)
                    writer_filtered_raw.writerow(next(csv_reader))
                    for line in csv_reader:
                        # 7 is the index number for the age in the file "complete_pl_output"
                        try:
                            if lower_age <= int(line[7]) <= upper_age:
                                # 9 is the index number for the description of the book
                                filtered_line = Llama_Test.Cleanup_Description(line[9])
                                writer_descriptions.writerow([filtered_line])
                                writer_filtered_raw.writerow(line)
                        except:
                            print("Error while filtering:\n" + line[9])


    def Recommend_for_Sample(sample_file, output_file):
        with open(sample_file) as inf:
            descriptions = inf.read()
        with open("API_Key.txt") as api_file:
            key = api_file.read()
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            # Options: llama-3.3-70b-versatile, mixtral-8x7b-32768, mistral-saba-24b
            model="gemma2-9b-it",
            messages=[
                {
                    "role": "user",
                    # Previous Description: "For all of the following list of book descriptions, without looking up the corresponding book or recommended age online, print 5 book recommendations followed by the given description on each line (of the format description:...description here...), separated by commas (ex: book_title_1, book_title_2,...book_title_5, description...and go on to the next line) for someone who read the book of the given description. Do this for ALL descriptions, even though the list is long. I'm writing this to a csv file. Output nothing else as it might mess up my program when I go in and read the csv file. Here are the descriptions: " + descriptions
                    "content": "For each of the following book descriptions, without looking up the corresponding book or recommended age online, print a 6 column csv file with the first five columns being five book recommendations for another reader of the same age as the one who enjoyed that book, and the sixth column being the original book description. (Example of one line: Book Recommendation 1, Book Recommendation 2, Book Recommendation 3, Book Recommendation 4, Book Recommendation 5, Original Book description). Here are the descriptions: " + descriptions
                },
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        with open(output_file, 'w') as of:
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    of.write(chunk.choices[0].delta.content)

    def Grab_Published_Description(title):
        """
        The purpose of this program is to take a book title, search the internet, and grab the book's published description, so that we can run the emotion vector calculator on
        each of the recommendations that the LLM gives us for each sample description.
        """
            # Google Books API endpoint
        url = f"https://www.googleapis.com/books/v1/volumes?q={title}"

        # Sending GET request to the Google Books API
        response = requests.get(url)

        # If the response is successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            
            # Check if any items are returned
            if "items" in data:
                book = data["items"][0]  # Taking the first result (you can handle multiple results if needed)
                
                # Extract description
                description = book["volumeInfo"].get("description", "No description available")
                
                return description
            else:
                return "No books found for that title."
        else:
            return f"Error: Unable to fetch data (Status code {response.status_code})"
        
    def Test_LLM_Recommendations(Recommendation_File):
        """
        The purpose of this function is to take in the LLM_Recommendations file, calculate the emotion vector for the sample description given,
        calculate the emotion vectors for each recommendation, and then take an average of these emotion vectors for final analysis.
        """
        pass

    def isStopWord(self, word):
        if word in self.stops:
            return True
        else:
            return False

    def isWordInEmotionFile(self, word):
        # Slightly faster implementation
        for key in self.emotion_dic.keys():
            if key.startswith(word + "#"):
                return True
        return False

    #function that get the emotion itensity
    def getEmotionItensity(self, word, emotion):
        key = word + "#" + emotion
        try:
            return self.emotion_dic[key]
        except:
            return 0.0

    #Assign the emotion itensity to the dictionary
    def calculateEmotion(self, emotions, word):
        emotions["Anger"] += self.getEmotionItensity(word, "anger")
        emotions["Anticipation"] += self.getEmotionItensity(word, "anticipation")
        emotions["Disgust"] += self.getEmotionItensity(word, "disgust")
        emotions["Fear"] += self.getEmotionItensity(word, "fear")
        emotions["Joy"] += self.getEmotionItensity(word, "joy")
        emotions["Sadness"] += self.getEmotionItensity(word, "sadness")
        emotions["Surprise"] += self.getEmotionItensity(word, "surprise")
        emotions["Trust"] += self.getEmotionItensity(word, "trust")

    #get the emotion vector of a given text
    def getEmotionVector(self, text, removeObj = False, useSynset = True):
        #create the initial emotions
        emotions = {"Anger": 0.0,
                    "Anticipation": 0.0,
                    "Disgust": 0.0,
                    "Fear": 0.0,
                    "Joy": 0.0,
                    "Sadness": 0.0,
                    "Surprise": 0.0,
                    "Trust": 0.0,
                    "Objective": 0.0}
        #parse the description
        str = re.sub("[^a-zA-Z]+", " ", text) # replace all non-letters with a space
        pat = re.compile(r'[^a-zA-Z ]+')
        str = re.sub(pat, '', str).lower() #  convert to lowercase

        #split string
        splits = str.split()
        
        #iterate over words array
        for split in splits:
            if not self.isStopWord(split):
                #first check if the word appears as it does in the text
                if self.isWordInEmotionFile(split): 
                    self.calculateEmotion(emotions, split)
                    
                # check the word in noun form (bats -> bat)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split)):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split))
                    
                # check the word in verb form (ran/running -> run)
                elif self.isWordInEmotionFile(self.lmtzr.lemmatize(split, 'v')):
                    self.calculateEmotion(emotions, self.lmtzr.lemmatize(split, 'v'))  
                    
                # check synonyms of this word
                elif useSynset and wordnet.synsets(split) is not None:
                    # only check the first two "senses" of a word, so we don't stray too far from its intended meaning
                    for syn in wordnet.synsets(split)[0:1]:
                        for l in syn.lemmas():
                            if self.isWordInEmotionFile(l.name()):
                                self.calculateEmotion(emotions, l.name())
                                continue
                                
                    # none of the synonyms matched something in the file
                    emotions["Objective"] += 1
                    
                else:
                    # not found in the emotion file, assign a score to Objective instead
                    emotions["Objective"] += 1

        # remove the Objective category if requested
        if removeObj:
            del emotions['Objective']
            
        total = sum(emotions.values())
        for key in sorted(emotions.keys()):
            try:
                # normalize the emotion vector
                emotions[key] = (1.0 / total) * emotions[key]
            except:
                emotions[key] = 0

        return emotions
    
    def Convert_to_Vectors(input_file, output_csv):
        """
        The purpose of this function is to take in the LLM recommendations that we've got and to assign an emotion vector
        to each of the book recommendations given, along with the original description given as input. This is in hopes of
        being able to better analyze how well the LLM did in its recommendations.
        """
        nltk.download('wordnet')
        with open(input_file) as inf:
            with open(output_csv,'w', encoding='utf-8') as of:
                writer = csv.writer(of)
                writer.writerow(["Book 1","Book 2","Book 3","Book 4", "Book 5","Original Description"])
                lines = inf.readlines()
                for line in lines:
                    if line.strip():
                        inputs = line.split(',')
                        outputs = []
                        filler_object = Llama_Test()
                        books = inputs[:5]
                        original_input = inputs[-1]
                        for book in books:
                            book_description = Llama_Test.Grab_Published_Description(book.strip())
                            outputs.append(Llama_Test.getEmotionVector(filler_object, book_description))
                        outputs.append(Llama_Test.getEmotionVector(filler_object, original_input.lstrip("description: ")))
                        writer.writerow(outputs)
                    
    def Calculate_Vector_Distance(vector_1, vector_2):
        """
        The purpose of this program is to take in two dictionary emotion vectors and to calculate the distance between the two
        in order to determine how well the LLM makes its recommendations relative to the original description.
        """
        distance = 0
        for key in vector_1:
            distance += (vector_2[key] - vector_1[key])**2

        return math.sqrt(distance)


    def Determine_Data_Difference(vector_csv, output_file):
        """
        The purpose of this function is to take in a file of the calculated vectors, from both the LLM recommendations and the
        original description, and to analyze the difference between the vector average of the recommendations with the original
        description. This is in hopes of noting statistical significance between the difference and coming to a scientific
        conclusion.
        """
        with open(vector_csv) as inf:
            with open(output_file, 'w', encoding='utf-8') as of:
                reader = csv.reader(inf)
                next(reader)
                for row in reader:
                    if row:
                        vectors = [ast.literal_eval(entry) for entry in row]
                        average_recommendation_vector = vectors[0]
                        for vector in vectors[1:5]:
                            for key in vector:
                                average_recommendation_vector[key] += vector[key]

                        for key in average_recommendation_vector:
                            average_recommendation_vector[key] /= 5

                        distance = Llama_Test.Calculate_Vector_Distance(average_recommendation_vector, vectors[-1])
                        
                        of.write(f"Average Recommendation Vector: {average_recommendation_vector}")
                        of.write(f"Input Description vector: {vectors[-1]}\n")
                        of.write(f"Distance between emotion vectors: {distance}\n\n")


        
            
    

            
# print(Llama_Test.Cleanup_Description(r"One thousand years after a cataclysmic event leaves humanity on the brink of extinction, the\n survivors take refuge in continuums designed to sustain the human race until repopulation of Earth becomes possible. Against this backdrop, a group of yinuums designed to sustain the human race until repopulation of Earth becomes possible. Against this backdrop, a group of young friends in \nthe underwater Thirteenth Continuum dream about life outside their totalitarian existence, an idea that has been outlawed for centuries. When a shocking discovery turns the dream into a reality, they must decide if they will risk their own extinction to experience something no one has for generations, the Surface-- Provided by publisher."))
Llama_Test.Filter_Sample(r"C:\Users\natha\Programming\LLM_Research\LLM_Semantics\Code\complete_pl_output.csv", 12, 19)
Llama_Test.Recommend_for_Sample("Sample_Data.txt", "LLM_Recommendations.txt")
Llama_Test.Convert_to_Vectors("LLM_Recommendations.txt", "Calculated_Emotion_Vectors.csv")
Llama_Test.Determine_Data_Difference("Calculated_Emotion_Vectors.csv", "Results.txt")

