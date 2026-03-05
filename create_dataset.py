import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("shravankumar9892/image-colorization", path="dataset/", unzip=True)