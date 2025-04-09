from emotion_vector_functions import getEmotionVector, cleanString
import pandas as pd

if __name__ == "__main__":
    
    in_file = "./Data/Teenager_GoodReads.csv"
    out_file = "./Data/Teenager_GoodReads_Emotion.csv"
    
    print(f"Reading from {in_file}")
    
    table = pd.read_csv(in_file, encoding="utf8", names = ["Index","ISBN", "ID", "Title", "Author", "Description", "Average_Rating", "Age"])
    
    
    out_db = pd.DataFrame(columns = ["ISBN", "ID", "Title", "Author", "Description", "Average_Rating", "Age", "EmotionVector"])
    
    
    
    print(f"{len(table)} lines in the input file")
    # read the description for each book
    for index,line in table.iterrows():
        # skip the first line
        if index == 0:
            continue
        if index % 1000 == 0:
            print(f"Processed {index} lines")
        
        # #,ISBN,ID,title,authors,description,avg_rating,age
        
        # get the description of the book
        description = line["Description"]
        # skip lines without a description
        if description == "" or description is None or not isinstance(description, str):
            continue
        
        # clean everything that could have commas
        title = cleanString(line["Title"])
        authors = cleanString(line["Author"])
        description = cleanString(description)
        
        
        # get the emotion vector for the description
        try:
            emotion_vector = getEmotionVector(description)
            new_row = [line["ISBN"],
                        line["ID"],
                        title,
                        authors,
                        description,
                        line["Average_Rating"],
                        line["Age"],
                        emotion_vector]
                        
            out_db.loc[-1] = new_row
            out_db.index += 1  # shift index
        except:
            continue
        
    # done looping
    print(f"Done. Output file has {len(out_db)} lines")
    
    # save the database to a csv file
    print(f"Saving to {out_file}")
    out_db.to_csv(out_file, index=False, encoding="utf8")