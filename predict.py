import torch
from collections import defaultdict 

tool_usage ={"Empty" : "T0",
            "Needle_driver": "T1",
            "Forceps": "T2",
            "Scissors":"T3"}
names = {int(v[-1]):k for k,v in tool_usage.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(path):
    model = torch.hub.load('our_yolov5', 'custom', path=fr"best.pt", source='local') 
    model.max_det = 2
    model = model.to(device)
    model.eval()
    results = model(path)
    name = path.split('.')[0].split('/')[-1]
    results.show()
    results.save()
    df = results.pandas().xyxy[0]

    print(df)


if __name__ == '__main__':
    path = './datasets/HW1_dataset/images/P025_balloon1_1679.jpg'
    main(path)