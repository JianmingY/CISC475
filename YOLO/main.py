

if __name__ == '__main__':

    import torch
    import torchvision
    from IPython import display

    display.clear_output()

    import ultralytics

    ultralytics.checks()

    if torch.cuda.is_available():
        device = "cuda"
        print("Yes!")
    else:
        device = "cpu"
        print("No!")
    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')
    results = model.train(data="config.yaml", epochs=2)
