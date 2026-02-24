# :earth_americas: Word Ladder Search Algorithms

Classical word ladder problems transform one word into another using discrete character substitutions. In this assignment, the problem is generalized to operate in a continuous semantic embedding space using pretrained GloVe word vectors.
Each word in the vocabulary is represented as a 100-dimensional vector in semantic space. The objective is to search for a sequence of semantically related words that transforms a start word into a goal word using classical AI search algorithms.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run app.py
   ```
