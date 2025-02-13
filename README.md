# Training LLM's for Accurate Book Recommendations

This project is aimed at understanding how different age groups develop preferences for different types of books and training large language models to more accurately recommend an age group for any given book. The project accomplishes this aim by accessing large data sets of book descriptions, analyzing the language used in the description, and assigning an emotion vector to each book. These emotion vectors are then compared to preferences of different age groups found by a certain study to recommend the book to a group.

## Getting Started

These instructions will help you prepare your environment and run the project on your personal device

### Prerequisites

The things you need before installing the software.

* Libraries to install
    * pandas
        * Provides an organized way to display information such as vector deta for any given book
    * nltk
        * A toolkit that helps analyze natural human language
    * html2text
        * Extract info from HTML while keeping the structure
    * lxml
        * Used for parsing, processing, and manipulating HTML documents efficiently
    * numpy
        * Used for numerical computation
    * requests
        * Allows for HTTP requests within a Python program
    * googlesearch
        * Allows for google searches to be made and fetched within a Python program
    * spacy
        * Allows for text analysis with large amounts of text
    * sklearn
        * Machine learning library that provides tools for data mining, analysis, and predictive modeling