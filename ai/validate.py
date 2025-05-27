import argparse
import torch
import numpy as np
from ai.data.datasets import AVLip
import torch.utils.data
#from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from tqdm import tqdm
from ai.trainer.trainer import Trainer


def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, crops, label in tqdm(loader):
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true = np.array(y_true)
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)

    if np.array_equal(y_true, y_pred):
        return 1.0, 0.0, 0.0, 1.0
    else:
        # Get AP
        ap = average_precision_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tp, fn, fp, tn = cm.ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        acc = accuracy_score(y_true, y_pred)

        return ap, fpr, fnr, acc
    
def test(model, loader):
    print("testing...")
    count_real = 0
    count_fake = 0
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, crops in tqdm(loader):
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)
    for i, y in enumerate(y_pred):
        if y == 0:
            count_real += 1
        else:
            count_fake += 1
    print("count_real",count_real)
    print("count_fake",count_fake)
    if count_real > count_fake:
        return True
    else:
        return False


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--real_list_path", type=str, default="/kaggle/input/deepfake-val-test/LipFD/datasets/0_real")
    # parser.add_argument("--fake_list_path", type=str, default="/kaggle/input/deepfake-val-test/LipFD/datasets/1_fake")
    # parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")
    # parser.add_argument("--batch_size", type=int, default=10)
    # parser.add_argument("--data_label", type=str, default="val")
    # parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    # parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt.pth")
    # parser.add_argument("--gpu", type=int, default=0)

    # opt = parser.parse_args()

    # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using cuda {opt.gpu} for inference.")

    model = Trainer()
    # state_dict = torch.load(opt.ckpt, map_location="cpu")
    # model.load_state_dict(state_dict["model"])
    # print("Model loaded.")
    # model.eval()
    # model.to(device)

    # dataset = AVLip(opt)
    # loader = data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=opt.batch_size, shuffle=True
    # )
    # ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])
    # print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")
    all_data = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/WEBDF/WebFAPI/static/preprocessing"
    dataset = AVLip(all_data)
    #print(dataset.__len__())
    data_loader = torch.utils.data.DataLoader(  
    dataset, batch_size=10, shuffle=False
    )
    model = Trainer()
    model.to("cuda:1")
    res = test(model.model, data_loader)