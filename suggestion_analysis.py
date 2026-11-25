import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st
df = pd.read_csv("c:/Users/Chirag Patil/OneDrive/Desktop/gauri/field project/FP_data2.csv")
print(df.columns)
def suggestion_plot():
    text = df["Suggestions"]
    text = text.dropna()
    text = text.str.lower()
    text = text.astype(str)
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    print(text.head(10))
    words = " ".join(text).split()
    stopwords = {
        "the", "is", "and", "to", "of", "a",
        "in", "for", "no", "with", "can", "use","be","not"}
    filtered_words = [
        word for word in words
        if word.lower() not in stopwords]
    print(filtered_words[:20])
    word_count = Counter(filtered_words)
    print(word_count.most_common(10))
    common_words = word_count.most_common(10)
    words_list = [item[0] for item in common_words]
    counts = [item[1] for item in common_words]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(words_list, counts)
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Most Frequent Words in Survey Responses")
    plt.xticks(rotation=45)
    return filtered_words, common_words, fig, word_count