import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from trainer import Trainer
from data import AVLip, create_dataloader
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image

def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []

    fn_count = 0
    fp_count = 0
    max_save = 3  # số ảnh tối đa cần lưu cho mỗi loại

    with torch.no_grad():
        for img, crops, label, paths in tqdm(loader):  # paths = list of strings
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            label = label.to(device).float()

            model.set_input((img_tens, crops_tens, label))
            model.forward()
            outputs = model.output.sigmoid().flatten()
            preds = (outputs >= 0.5).float()

            y_pred.extend(preds.tolist())
            y_true.extend(label.flatten().tolist())

            for i in range(len(label)):
                true = label[i].item()
                pred = preds[i].item()
                path = paths[i]

                # FN: fake (1) → predicted real (0)
                if true == 1 and pred == 0 and fn_count < max_save:
                    img_pil = Image.open(path).convert("RGB")
                    img_pil.save(f"fnr_example_raw_{fn_count}.png")
                    print(f"🟠 Saved FN #{fn_count} from: {path}")
                    fn_count += 1

                # FP: real (0) → predicted fake (1)
                if true == 0 and pred == 1 and fp_count < max_save:
                    img_pil = Image.open(path).convert("RGB")
                    img_pil.save(f"fpr_example_raw_{fp_count}.png")
                    print(f"🔵 Saved FP #{fp_count} from: {path}")
                    fp_count += 1

                if fn_count >= max_save and fp_count >= max_save:
                    break

            if fn_count >= max_save and fp_count >= max_save:
                break

        loss = model.get_loss()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.array_equal(y_true, y_pred):
        return 1.0, 0.0, 0.0, 1.0, loss
    else:
        ap = average_precision_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tp, fn, fp, tn = cm.ravel()
        fnr = fn / (fn + tp + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        acc = accuracy_score(y_true, y_pred)
        return ap, fpr, fnr, acc, loss



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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()

    device_str = f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu"
    loader = load_data(opt.real_path, opt.fake_path, opt.batch_size)

    model = Trainer(
        checkpoints_dir="",
        save_name="",
        lr=2e-4,
        optim="adam",
        fix_encoder=True,
        load_checkpoint="./checkpoints/mobilenet_server/model_epoch_17.pth"
    )

    model.eval()
    model.to(device_str)

    ap, fpr, fnr, acc, loss = validate(model, loader, device_str)
    print(f"\n✅ acc: {acc:.4f} | ap: {ap:.4f} | fpr: {fpr:.4f} | fnr: {fnr:.4f} | loss: {loss:.4f}")
