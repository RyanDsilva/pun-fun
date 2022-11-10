import nltk

def setup():
  nltk.download('punkt', quiet=True)
  nltk.download('stopwords', quiet=True)
  nltk.download('wordnet', quiet=True)
  nltk.download('omw-1.4', quiet=True)
  nltk.download('cmudict', quiet=True)
  nltk.download('averaged_perceptron_tagger', quiet=True)
  print('[INFO] Setup completed successfully.')