echo "Downloading models from kaggle..."

python scripts/download_model.py

echo "Downloading NLTK resources..."

python -c "
import nltk
import os

# Set the data path
# Download with error handling
packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
for package in packages:
    try:
        nltk.download(package, quiet=False)
        print(f'Successfully downloaded {package}')
    except Exception as e:
        print(f'Error downloading {package}: {e}')
        # Try default location as fallback
        try:
            nltk.download(package, quiet=False)
            print(f'Downloaded {package} to default location')
        except Exception as e2:
            print(f'Failed to download {package}: {e2}')
"

echo "Verifying NLTK data..."
python -c "
import nltk

try:
    from nltk.corpus import stopwords
    print('✓ Stopwords accessible')
    print(f'Available languages: {stopwords.fileids()[:5]}...')
except Exception as e:
    print(f'✗ Stopwords not accessible: {e}')
"
