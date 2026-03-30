import os
import torch 
from tqdm import tqdm
from model_pretrained import YOLOV1
from metrics import loss
from dataset import train_loader, test_loader
from metrics import mAP
from utils import get_bboxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

#Hyperparameter
epochs = 70
weight_decay = 2e-4

def train():
    model = YOLOV1().to(device)
    criterion = loss
    best_map = 0.0
    start_epoch = 0
    checkpoint_file = "yolo_checkpoint_pretrained.pth"

    if os.path.exists(checkpoint_file):
        print(f"-> Found checkpoint '{checkpoint_file}'. Loading...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_map"]
        print(f"Resuming {start_epoch + 1}")
    else:
        print("Starting from scratch")

    print("\nStarting training process...")

    scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu')

    if start_epoch <= 10:
        model.freeze_weights()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        model.unfreeze_weights()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    #Training loop
    for epoch in range(start_epoch, epochs):
        if epoch == 11:
            model.unfreeze_weights()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for idx, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu'):
                outputs = model(inputs)
                _, _, _, _, batch_loss = criterion(labels, outputs)
            
            scaler.scale(batch_loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += batch_loss.item() * labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=batch_loss.item())
    
        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        print(f"\n--- Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} ---") 
        val_loss, map_val = val(model, criterion)
        print(f"Validation Loss: {val_loss:.4f}, mAP: {map_val:.4f}")
        
        if map_val > best_map:
            best_map = map_val
            print(f"--> Overwrote best model with mAP: {best_map:.4f}")
            torch.save(model.state_dict(), "best_yolo_model.pth")
            
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": best_map,
        }
        torch.save(checkpoint, checkpoint_file)
    
def val(model, criterion):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, _, _, _, batch_loss = criterion(labels, outputs)
            test_loss += batch_loss.item() * inputs.size(0)

    test_loss_avg = test_loss / len(test_loader.dataset)
    
    pred_boxes, true_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.05, device=device
    )
    
    map_val = mAP(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
    )

    model.train()

    return test_loss_avg, map_val.item()

if __name__ == "__main__":
    train()
