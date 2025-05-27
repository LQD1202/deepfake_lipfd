import torch
from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from data.datasets import AVLip
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

if __name__ == "__main__":  # 🔥 QUAN TRỌNG: Bọc toàn bộ code trong đây!


    real_path = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/lip_av/mobilenetv3.pytorch/datasets/0_real"
    fake_path = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f4/Dat/lip_av/mobilenetv3.pytorch/datasets/1_fake"

    real_imgs = [(os.path.join(real_path, f), 0) for f in sorted(os.listdir(real_path))]
    fake_imgs = [(os.path.join(fake_path, f), 1) for f in sorted(os.listdir(fake_path))]

    all_data = real_imgs + fake_imgs
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)
    #print("data is split")
    # Create dataloaders
    train_dataset = AVLip(train_data)
    val_dataset = AVLip(val_data)

    train_loader = create_dataloader(train_dataset, batch_size=10, is_train=True)
    val_loader = create_dataloader(val_dataset, batch_size=10, is_train=False)
    print("data is ready")
    # Initialize model
    model = Trainer()
    #model.load_networks("model_epoch_0.pth")

    # Print dataset sizes
    print("Length of data loader: %d" % len(train_loader))
    print("Length of val loader: %d" % len(val_loader))

    for epoch in range(10):
        model.train()
        print("epoch: ", epoch + model.step_bias)
        for i, (img, crops, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.total_steps += 1

            # Ensure all input data is on the correct device
            # print("[DEBUG] img device:", img.device)

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()
            model.optimize_parameters()
            # print("[DEBUG] model device:", model.device)
            if model.total_steps % 100 == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )

        print("loss: ", loss)

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, "cuda" if torch.cuda.is_available() else "cpu")
        if acc < last_acc:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" % (epoch+ + model.step_bias))
            last_acc = acc
        acc = [ap, fpr, fnr, acc]
        model.log_model(epoch, loss, acc)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
