echo "Downloading models from kaggle..."

python scripts/download_model.py

echo "Downloading NLTK resources..."

python -c "import nltk; nltk.download('stopwords', download_dir='/opt/render/nltk_data'); nltk.download('punkt', download_dir='/opt/render/nltk_data'); nltk.download('wordnet', download_dir='/opt/render/nltk_data'); nltk.download('omw-1.4', download_dir='/opt/render/nltk_data')"
