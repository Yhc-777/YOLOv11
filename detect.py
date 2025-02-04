from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model='/media/u/bak/6t/haochen/YOLOv11/runs/train/exp/weights/best.pt')  
    model.predict(source='/media/u/bak/6t/haochen/YOLOv11/datasets/cigarette/valid/images',
                  save=True,
                  show=True,
                  )