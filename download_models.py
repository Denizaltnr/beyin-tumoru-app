import gdown
import os

def download_model(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"📥 {output_path} indiriliyor...")
    gdown.download(url, output_path, quiet=False)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # Örnek: CNN modeli (Google Drive ID'sini kendin koyman gerek)
    download_model("1fSVLr3PjR7YsUpBKgU4MvqxG9YzvuN6k", "models/brain_tumor_cnn_model.h5")

    # Diğer modeller (Google Drive ID’lerini sen eklemelisin)
    download_model("1CDPS2wtC8BTFeCWQQn5htIi9lms4ROk9", "models/mobilenet_brain_tumor_model.h5")
    download_model("18aoVfmxPaLGV902UulBhQKNrqHEt1TVt", "models/vgg16_brain_tumor_model.h5")
    download_model("1PUysW4CWS69HnAOTZTkENOhdi56Bm7dt", "models/svm_model.pkl")
    download_model("1sJq3lUssRlZrxUK6fFkj9U4WhY9z1BRp", "models/random_forest.npy")
