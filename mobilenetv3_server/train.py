import torch
from validate import validate
from data import create_dataloader
from trainer import Trainer
from data.datasets import AVLip
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from newmodel import extract_tinyclip_feature
import os

if __name__ == "__main__":  # 🔥 QUAN TRỌNG: Bọc toàn bộ code trong đây!

    # Paths to training data
    real_path = 
    fake_path = 

    # Labeling images
    real_imgs = [(os.path.join(real_path, f), 0) for f in sorted(os.listdir(real_path))]
    fake_imgs = [(os.path.join(fake_path, f), 1) for f in sorted(os.listdir(fake_path))]

    # Combine training data
    train_data = real_imgs + fake_imgs

    # Paths to validation data
    real_path_val = 
    fake_path_val = 

    # Labeling validation images
    real_imgs_val = [(os.path.join(real_path_val, f), 0) for f in sorted(os.listdir(real_path_val))]
    fake_imgs_val = [(os.path.join(fake_path_val, f), 1) for f in sorted(os.listdir(fake_path_val))]

    # Combine validation data
    val_data = real_imgs_val + fake_imgs_val

    # Create datasets and dataloaders
    train_dataset = AVLip(train_data)
    val_dataset = AVLip(val_data)

    train_loader = create_dataloader(train_dataset, batch_size=40, is_train=True)
    val_loader = create_dataloader(val_dataset, batch_size=40, is_train=False)

    #print("data is ready")
    # Initialize model
    model = Trainer(
                checkpoints_dir="./checkpoints",
                save_name="mobilenet_server",
                lr=2e-4,
                optim="adam",
                fix_encoder=True,
                load_checkpoint=""
            )

    # Print dataset sizes
    print("Length of data loader: %d" % len(train_loader))
    print("Length of val loader: %d" % len(val_loader))

    for epoch in range(20):
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
        if epoch % 1 == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" % (epoch + model.step_bias))

        model.eval()
        ap, fpr, fnr, acc, loss_val = validate(model, val_loader, "cuda:1" if torch.cuda.is_available() else "cpu")
        model.log_model(epoch, loss, acc, loss_val)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
