import os
import urllib.request
import gzip
import shutil
import numpy as np

def download_mnist(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        'train-images.idx3-ubyte': 'train-images-idx3-ubyte.gz',
        'train-labels.idx1-ubyte': 'train-labels-idx1-ubyte.gz',
        't10k-images.idx3-ubyte': 't10k-images-idx3-ubyte.gz',
        't10k-labels.idx1-ubyte': 't10k-labels-idx1-ubyte.gz'
    }
    
    for dest_name, gz_name in files.items():
        dest_path = os.path.join(target_dir, dest_name)
        gz_path = os.path.join(target_dir, gz_name)
        
        if not os.path.exists(dest_path):
            print(f"File {dest_name} not found. Downloading from {base_url + gz_name}...")
            url = base_url + gz_name
            try:
                urllib.request.urlretrieve(url, gz_path)
                print(f"Downloaded {gz_name}. Extracting...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(dest_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)
                print(f"Extracted and saved {dest_name}.")
            except Exception as e:
                print(f"Error downloading {gz_name} from {url}: {e}")
                # Try fallback mirror
                fallback_url = f"https://github.com/cvdf-datasets/mnist/raw/master/{gz_name}"
                print(f"Trying fallback: {fallback_url}")
                try:
                    urllib.request.urlretrieve(fallback_url, gz_path)
                    print(f"Downloaded from fallback {gz_name}. Extracting...")
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(dest_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(gz_path)
                    print(f"Extracted and saved {dest_name}.")
                except Exception as e_fallback:
                    raise RuntimeError(f"Failed to download MNIST data from both primary and fallback sources: {e_fallback}")

def load_mnist(data_dir=None):
    """
    Load MNIST data.
    If data_dir is not provided or files do not exist, downloads it automatically.
    """
    # If data_dir is not provided, look in local project path
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'data', 'mnist')
        
    print(f"Checking for MNIST data in: {data_dir}")
    
    required_files = [
        'train-images.idx3-ubyte',
        'train-labels.idx1-ubyte',
        't10k-images.idx3-ubyte',
        't10k-labels.idx1-ubyte'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing_files:
        print(f"Missing files in {data_dir}: {missing_files}")
        # If it's the kaggle directory, we can't write there. Change to local path.
        if "/kaggle" in data_dir:
            print("Cannot write to Kaggle path. Switching to local directory: ./data/mnist")
            data_dir = os.path.join(os.getcwd(), 'data', 'mnist')
            # Check again if missing
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
            
        if missing_files:
            download_mnist(data_dir)
            
    # Load raw binary data
    def load_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28).astype(np.float32)
    
    def load_labels(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
        
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    print("Loading training data...")
    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    
    print("Loading test data...")
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)
    
    # Combine train and test
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    print(f"Total images loaded: {len(X)}")
    return X, y
