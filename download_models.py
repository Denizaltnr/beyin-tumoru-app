import gdown
import os

os.makedirs("model", exist_ok=True)

files = {
    "svm_model.pkl": "1PUysW4CWS69HnAOTZTkENOhdi56Bm7dt",
    "mobilenet_brain_tumor_model.h5": "1CDPS2wtC8BTFeCWQQn5htIi9lms4ROk9",
    "vgg16_brain_tumor_model.h5": "18aoVfmxPaLGV902UulBhQKNrqHEt1TVt",
    "brain_tumor_cnn_model.h5": "1fSVLr3PjR7YsUpBKgU4MvqxG9YzvuN6k"
}

for name, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join("model", name)
    print(f"⏬ {name} indiriliyor...")
    gdown.download(url, output, quiet=False)
