import nltk

def setup():
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  nltk.download('cmudict')
  nltk.download('averaged_perceptron_tagger')
  print('[INFO] Setup completed successfully.')