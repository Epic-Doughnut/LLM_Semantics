from emotion_vector_functions import getEmotionVector
import pandas as pd

if __name__ == "__main__":
    
    in_file = "./Data/Teenager_GoodReads.csv"
    out_file = "./Data/Teenager_GoodReads_Emotion.csv"
    
    table = pd.read_csv(in_file, encoding="utf8", names = ["Index","ISBN", "ID", "Title", "Author", "Description", "Average_Rating", "Age"])
    
    
    
    res = open(out_file, "w", encoding="utf8", errors="ignore")
    
    # write the header of the csv file
    res.write("ISBN,ID,Title,Author,Description,Average_Rating,Age,EmotionVector\n")
    # open Teenager_GoodReads.csv
    
    
    # read the description for each book
    for index,line in table.iterrows():
        # skip the first line
        if index == 0:
            continue
        
        # #,ISBN,ID,title,authors,description,avg_rating,age
        # get the description of the book
        description = line["Description"]
        # skip lines without a description
        if description == "" or description is None or not isinstance(description, str):
            continue
        # print(description)
        # get the emotion vector for the description
        try:
            emotion_vector = getEmotionVector(description)
            res.write(f"{line["ISBN"]},{line["ID"]},{line["Title"]},{line["Author"]},\"{description}\",{line["Average_Rating"]},{line["Age"]},{emotion_vector}\n")
        except:
            continue
        