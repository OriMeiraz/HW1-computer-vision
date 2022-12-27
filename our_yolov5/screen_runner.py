import os

#for bs in [16, 64, 128]:
#	os.system(f"screen python train.py --img 640 --batch {bs} --epochs 20 --data HW1_dataset.yaml --weights yolov5m.pt")
#for lr in [0.0005, 0.0010, 0.0050]:
#	os.system(f"screen python train.py --img 640 --batch 16 --epochs 20 --data HW1_dataset.yaml --weights yolov5m.pt --my_lr0 {lr}")
for optimizer in ['AdamW']:
	os.system(f"screen python train.py --img 640 --batch 16 --epochs 20 --data HW1_dataset.yaml --weights yolov5m.pt --optimizer Adam; python train.py --img 640 --batch 16 --epochs 20 --data HW1_dataset.yaml --weights yolov5m.pt --optimizer AdamW")
#python screen_runner.py