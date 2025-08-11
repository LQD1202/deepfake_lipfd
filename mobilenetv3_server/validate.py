import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from trainer import Trainer
from data import AVLip, create_dataloader
#from newmodel import extract_tinyclip_feature
import torch.nn as nn

def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, crops, label in tqdm(loader):
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            label = label.to(device).float()

            model.set_input((img_tens, crops_tens, label))
            model.forward()
            outputs = model.output.sigmoid().flatten().tolist()
            y_pred.extend(outputs)
            y_true.extend(label.flatten().cpu().tolist())
        loss = model.get_loss()
    y_true = np.array(y_true)
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)

    if np.array_equal(y_true, y_pred):
        return 1.0, 0.0, 0.0, 1.0
    else:
        ap = average_precision_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tp, fn, fp, tn = cm.ravel()
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        acc = accuracy_score(y_true, y_pred)

        return ap, fpr, fnr, acc, loss

def test(trainer_model, loader, gpu_id):
    print("Running validation...")
    device = torch.device(gpu_id)
    model = trainer_model.model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for img, crops, label in tqdm(loader):
            img = img.to(device)
            crops = [[c.to(device) for c in crop_set] for crop_set in crops]

            features = extract_tinyclip_feature(img).to(device)
            outputs = model(crops, features)[0].sigmoid().flatten()

            y_pred.extend(outputs.cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred_binary = np.array(y_pred) >= 0.5

    if np.array_equal(y_true, y_pred_binary):
        return 1.0, 0.0, 0.0, 1.0

    ap = average_precision_score(y_true, y_pred_binary)
    cm = confusion_matrix(y_true, y_pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
    else:
        fpr = fnr = 0.0

    acc = accuracy_score(y_true, y_pred_binary)
    return ap, fpr, fnr, acc


def load_data(real_path, fake_path, batch_size):
    real_imgs = [(os.path.join(real_path, f), 0) for f in sorted(os.listdir(real_path))]
    fake_imgs = [(os.path.join(fake_path, f), 1) for f in sorted(os.listdir(fake_path))]
    all_data = real_imgs + fake_imgs

    dataset = AVLip(all_data)
    loader = create_dataloader(dataset, batch_size=batch_size, is_train=False)
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, default="./val/0_real")
    parser.add_argument("--fake_path", type=str, default="./val/1_fake")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/mobilenet_kaggle")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_file", type=str, default="results.csv")

    opt = parser.parse_args()

    device_str = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"
    loader = load_data(opt.real_path, opt.fake_path, opt.batch_size)

    with open(opt.results_file, "w") as f:
        f.write("Checkpoint,Accuracy,AP,FPR,FNR\n")

        for ckpt_name in sorted(os.listdir(opt.checkpoints_dir)):
            if not ckpt_name.endswith(".pth"):
                continue
            ckpt_path = os.path.join(opt.checkpoints_dir, ckpt_name)
            print(f"\n🔍 Evaluating checkpoint: {ckpt_name}")

            model = Trainer(
                checkpoints_dir=opt.checkpoints_dir,
                save_name="mobilenet_kaggle",
                lr=2e-4,
                optim="adam",
                fix_encoder=True,
                load_checkpoint=ckpt_path
            )

            model.eval()
            model.to(device_str)

            ap, fpr, fnr, acc = validate(model, loader, device_str)
            print(f"✅ acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.4f} | fnr: {fnr:.4f}")

            f.write(f"{ckpt_name},{acc:.4f},{ap:.4f},{fpr:.4f},{fnr:.4f}\n")
