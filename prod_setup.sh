echo "Downloading models from kaggle..."

python scripts/download_model.py

echo "Downloading NLTK resources..."

python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
