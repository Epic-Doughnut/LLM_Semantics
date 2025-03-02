from groq import Groq
import csv
import requests

class Llama_Test:
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
                                writer_descriptions.writerow([line[9]])
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
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Given the following list of book descriptions, without looking up the corresponding book or recommended age online, print 5 book recommendations separated by commas (ex: book_title_1, book_title_2...book_title_5...and go on to the next line) for someone who read the book of the given description. Do this for ALL descriptions, even though the list is long. I'm writing this to a csv file. Here are the descriptions: " + descriptions
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
        with open(Recommendation_File) as input_file:
            lines = input_file.readlines()
            # This list comprehension is supposed to filter the fluff at the top and get rid of the new lines
            lines = [line for line in lines[2] if line.strip()]
            sample_description_to_recommendations = {}
            

print(Llama_Test.Grab_Published_Description("Percy Jackson and the Lightning Thief"))
# Llama_Test.Filter_Sample("..\complete_pl_output.csv", 12, 19)
# Llama_Test.Recommend_for_Sample("Sample_Data.txt", "LLM_Recommendations.txt")

