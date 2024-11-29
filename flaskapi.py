from flask import Flask, request, send_file, jsonify
from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import torch
import os
import shutil
from io import BytesIO
import tarfile

app = Flask(__name__)

# Set up folders and constants
upload_folder = 'inputs'
result_folder = 'results'
IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

# Ensure folders exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
torch.set_num_threads(12)
# Set device to CPU
device = torch.device('cpu')
print('Running on device:', device)

# Initialize RealESRGAN model on CPU with scale factor
model_scale = 2
model = RealESRGAN(device, scale=model_scale)
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')


@app.route("/", methods=["GET"])
def no_file_provided():
    return jsonify({"error": "No file provided"}), 400

def image_to_tar_format(img, image_name):
    buff = BytesIO()
    if '.png' in image_name.lower():
        img = img.convert('RGBA')
        img.save(buff, format='PNG')
    else:
        img.save(buff, format='JPEG')
    buff.seek(0)
    img_tar_info = tarfile.TarInfo(name=image_name)
    img_tar_info.size = len(buff.getvalue())
    return img_tar_info, buff


def process_tar(path_to_tar):
    processing_tar = tarfile.open(path_to_tar, mode='r')
    result_tar_path = os.path.join(result_folder, os.path.basename(path_to_tar))
    save_tar = tarfile.open(result_tar_path, 'w')

    for c, member in enumerate(processing_tar):
        print(f'{c}, processing {member.name}')

        if not member.name.endswith(IMAGE_FORMATS):
            continue

        try:
            img_bytes = BytesIO(processing_tar.extractfile(member.name).read())
            img_lr = Image.open(img_bytes, mode='r').convert('RGB')
        except Exception as err:
            print(f'Unable to open file {member.name}, skipping')
            continue

        img_sr = model.predict(np.array(img_lr))
        img_tar_info, fp = image_to_tar_format(img_sr, member.name)
        save_tar.addfile(img_tar_info, fp)

    processing_tar.close()
    save_tar.close()
    print(f'Finished! Archive saved to {result_tar_path}')
    return result_tar_path


def process_input(filename):
    if tarfile.is_tarfile(filename):
        return process_tar(filename)
    else:
        try:
            result_image_path = os.path.join(result_folder, os.path.basename(filename))
            image = Image.open(filename).convert('RGB')
            sr_image = model.predict(np.array(image))
            sr_image.save(result_image_path)
            print(f'Finished! Image saved to {result_image_path}')
            return result_image_path
        except Exception as e:
            raise Exception("The image could not be upscaled. Try a different image or re-upload later.") from e


@app.route("/process/", methods=["POST"])
def upscale_image():
    """Endpoint to process an uploaded image or tar archive."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    file_location = os.path.join(upload_folder, file.filename)

    try:
        file.save(file_location)
    except Exception as e:
        return jsonify({"error": "File upload failed"}), 500

    if file.filename.lower().endswith(IMAGE_FORMATS) or tarfile.is_tarfile(file_location):
        try:
            result_path = process_input(file_location)
            return send_file(result_path, as_attachment=True)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400


if __name__ == "__main__":
    app.run(port=5000)
