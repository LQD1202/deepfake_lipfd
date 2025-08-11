import argparse
import os
from data import AVLip, create_dataloader
from trainer import Trainer
import torch 
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score

def load_data(root_dir, batch_size):
    real_path = os.path.join(root_dir, "real")
    all_data = [(os.path.join(real_path, f), 0) for f in sorted(os.listdir(real_path))]

    for subdir in os.listdir(root_dir):
        fake_path = os.path.join(root_dir, subdir)
        if subdir == "real" or not os.path.isdir(fake_path):
            continue
        all_data += [(os.path.join(fake_path, f), 1) for f in sorted(os.listdir(fake_path))]

    dataset = AVLip(all_data)
    loader = create_dataloader(dataset, batch_size=batch_size, is_train=False)
    return loader

def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device("cuda")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:/datasets/AVLips")
    parser.add_argument("--checkpoints_dir", type=str, default="D:/checkpoints/mobilenetv3")
    parser.add_argument("--checkpoint_file", type=str, default="E:/shufflenet_v2_distulation/checkpoints/mobilenet_server/model_epoch_17.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()
    device_str = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"

    # Load data
    loader = load_data(opt.root_dir, opt.batch_size)

    # Load model
    model = Trainer(
        checkpoints_dir=opt.checkpoints_dir,
        save_name="",
        lr=2e-4,
        optim="adam",
        fix_encoder=True,
        load_checkpoint=os.path.join(opt.checkpoints_dir, opt.checkpoint_file)
    )

    model.eval()
    model.to(device_str)

    # Validate
    ap, fpr, fnr, acc, loss = validate(model, loader, device_str)
    print(f"✅ acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.4f} | fnr: {fnr:.4f} | loss: {loss:.4f}")
