from groq import Groq
import csv

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
                    "content": "Solely from reading the description (without looking up the book and the published age recommendation), recommend me 5 other book titles in the format (book title 1, book title 2, etc.) that would be suitable for a reader of the same estimated age as the one that read a book with each of the following descriptions:" + descriptions
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
    
Llama_Test.Filter_Sample("..\complete_pl_output.csv", 12, 19)
Llama_Test.Recommend_for_Sample("Sample_Data.txt", "LLM_Recommendations.txt")

