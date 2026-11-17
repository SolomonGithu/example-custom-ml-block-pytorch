import argparse, os, shutil, subprocess, sys
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data-directory', required=True)
parser.add_argument('--out-directory', required=True)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--model-size', default='n')
parser.add_argument('--batch-size', type=int, default=16)
#args = parser.parse_args()

parser.add_argument('--info-file', default=None)  # passed by Edge Impulse, not needed
args, unknown = parser.parse_known_args()
if unknown:
    print('Ignoring unrecognized arguments:', unknown)

here = os.path.dirname(os.path.abspath(__file__))

# 1. Convert Edge Impulse data to YOLO format (writes to /tmp/data inside the container)
cmd = [sys.executable, os.path.join(here, 'extract_dataset.py'),
       '--data-directory', args.data_directory,
       '--out-directory', '/tmp/data',
       '--model-size', args.model_size]
if args.info_file:
    cmd += ['--info-file', args.info_file]
subprocess.run(cmd, check=True)
with open('/tmp/data/image_size.txt') as f:
    image_size = int(f.read())

# 2. Fine-tune pretrained YOLO26
model = YOLO(f'/opt/weights/yolo26{args.model_size}.pt')
model.train(data='/tmp/data/data.yaml', imgsz=image_size, epochs=args.epochs,
            batch=args.batch_size, project='/tmp/runs', name='yolo26',
            exist_ok=True, plots=False, workers=2)

# 3. Export to ONNX
best = YOLO('/tmp/runs/yolo26/weights/best.pt')
onnx_path = best.export(format='onnx', imgsz=image_size, end2end=False)
os.makedirs(args.out_directory, exist_ok=True)
shutil.copy(onnx_path, os.path.join(args.out_directory, 'model.onnx'))

shutil.copy('/tmp/runs/yolo26/weights/best.pt', os.path.join(args.out_directory, 'best.pt'))
print('Done: model.onnx written')