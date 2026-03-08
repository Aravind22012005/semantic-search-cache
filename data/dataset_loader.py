from sklearn.datasets import fetch_20newsgroups

def load_dataset():

    data = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )

    documents = data.data
    labels = data.target

    cleaned_docs = []

    for doc in documents:
        if len(doc) > 50:   # remove extremely short docs
            cleaned_docs.append(doc)

    return cleaned_docs