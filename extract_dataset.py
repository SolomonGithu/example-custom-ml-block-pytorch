import numpy as np
import argparse, math, shutil, os, json, time
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse => YOLO')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--model-size', type=str, help='e.g. "n" for nano', required=True)
parser.add_argument('--info-file', type=str, default=None)
args = parser.parse_args()

# Images are in X_*.npy, labels are in Y_*.json (under "samples")
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

def load_labels(name):
    # labels may be stored as .json or .npy, as either {"samples": [...]} or a plain list
    for ext in ('.json', '.npy'):
        path = os.path.join(args.data_directory, name + ext)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.loads(f.read())
            return data['samples'] if isinstance(data, dict) else data
    print('Error: no label file found for', name, 'in', args.data_directory)
    exit(1)

Y_train = load_labels('Y_split_train')
Y_test = load_labels('Y_split_test')

if len(X_train) != len(Y_train) or len(X_test) != len(Y_test):
    print('Error: image count and label count differ',
          len(X_train), len(Y_train), len(X_test), len(Y_test))
    exit(1)

def infer_shape(n_features):
    # square images, either 1 channel (grayscale) or 3 channels (RGB)
    s = int(round(math.sqrt(n_features)))
    if s * s == n_features:
        return s, s, 1
    s = int(round(math.sqrt(n_features / 3)))
    if s * s * 3 == n_features:
        return s, s, 3
    print('Error: cannot work out a square image size from', n_features, 'features')
    exit(1)

if X_train.ndim == 2:
    image_width, image_height, image_channels = infer_shape(X_train.shape[1])
else:
    image_width, image_height, image_channels = list(X_train.shape[1:])
if image_width != image_height:
    print('Error: only square images are supported (width != height)')
    exit(1)
print('Image size:', image_width, 'x', image_height, 'channels:', image_channels)

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'image_size.txt'), 'w') as f:
    f.write(str(image_width))

class_count = 0
total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = round(time.time() * 1000)
converted_images = 0

def convert(X, Y, category):
    global class_count, last_printed, converted_images
    for ix in range(0, len(X)):
        if image_channels == 1:
            img_shape = (image_width, image_height)
        else:
            img_shape = (image_width, image_height, image_channels)

        raw_img_data = (np.reshape(X[ix], img_shape) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        images_dir = os.path.join(out_dir, category, 'images')
        labels_dir = os.path.join(out_dir, category, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        Image.fromarray(raw_img_data).save(os.path.join(images_dir, 'image' + str(ix).zfill(5) + '.jpg'))

        labels_text = []
        for l in labels:
            if l['label'] > class_count:
                class_count = l['label']
            x_center = (l['x'] + (l['w'] / 2)) / image_width
            y_center = (l['y'] + (l['h'] / 2)) / image_height
            width = l['w'] / image_width
            height = l['h'] / image_height
            # class x_center y_center width height (class ids start at 0)
            labels_text.append(str(l['label'] - 1) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))

        with open(os.path.join(labels_dir, 'image' + str(ix).zfill(5) + '.txt'), 'w') as f:
            f.write('\n'.join(labels_text))

        converted_images += 1
        if converted_images == 1 or round(time.time() * 1000) - last_printed > 3000:
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = round(time.time() * 1000)

convert(X=X_train, Y=Y_train, category='train')
convert(X=X_test, Y=Y_test, category='valid')
print('Converted', converted_images, 'images,', class_count, 'classes')

info_file = args.info_file or os.path.join(args.data_directory, 'train_input.json')
class_list = None
if os.path.exists(info_file):
    with open(info_file) as f:
        class_list = json.load(f).get('classes')
if not class_list:
    class_list = ['class' + str(c) for c in range(0, class_count)]
print('Classes:', class_list)

data_yaml = ('train: ' + os.path.join(os.path.abspath(out_dir), 'train', 'images') + '\n'
             'val: ' + os.path.join(os.path.abspath(out_dir), 'valid', 'images') + '\n\n'
             'nc: ' + str(len(class_list)) + '\n'
             'names: ' + json.dumps(class_list) + '\n')
with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)