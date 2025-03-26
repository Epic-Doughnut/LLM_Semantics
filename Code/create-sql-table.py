import MySQLdb as mysql
import pandas as pd

def create_db():
    db = establish_connection()
    c = db.cursor()
    c.execute("CREATE DATABASE IF NOT EXISTS books")
    db.close()

def establish_connection():
    mydb = mysql.connect(
        host="localhost",
        user="user",
        passwd="password"
    )
    
    return mydb

def clean_csv(path:str):
    
    # books = pd.read_csv(path + "Books.csv")
    ratings = pd.read_csv(path + "Ratings.csv")
    users = pd.read_csv(path + "Users.csv")
    
    # books = books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    
    # remove all the ratings of 0
    ratings = ratings[ratings['Book-Rating'] != 0]
    
    # remove all the users without age data
    users = users[pd.notna(users['Age'])]
    
    # remove all the users outside our age range (12-19)
    users = users[users['Age'] >= 12]
    users = users[users['Age'] <= 19]
    
    # remove all the users who have not rated any books
    users = users[users['User-ID'].isin(ratings['User-ID'])]
    
    # keep only the ratings of users which are in the users dataframe
    ratings = ratings[ratings['User-ID'].isin(users['User-ID'])]
    
    
    # write the cleaned data to a new file
    ratings.to_csv(path + "Ratings_cleaned.csv", index=False)
    users.to_csv(path + "Users_cleaned.csv", index=False)
    
    print("Done cleaning csv files")
    print(f"Total of {len(ratings)} ratings")
    print(f"Total of {len(users)} users")
        


if __name__ == '__main__':
    # clean_csv("../data/archive/")
    create_db()