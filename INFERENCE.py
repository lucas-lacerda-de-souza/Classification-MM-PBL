import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm
import csv


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="DLBCL vs ENKTCL Inference (Patch-level)")
parser.add_argument('--input_dir', type=str, required=True,
                    help="Directory containing image patches")
parser.add_argument('--output_dir', type=str, required=True,
                    help="Directory to save predictions")
parser.add_argument('--weights', type=str, required=True,
                    help="Path to model weights (.pth)")
parser.add_argument('--model', type=str, required=True,
                    choices=['alexnet', 'resnet50', 'densenet121',
                             'inceptionv3', 'cellvit'],
                    help="Model architecture")
args = parser.parse_args()


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------
def load_model(model_name, num_classes=2):
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes)

    elif model_name == "inceptionv3":
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "cellvit":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )

    else:
        raise ValueError("Unsupported model architecture.")

    return model


# ---------------------------------------------------------
# Device + Model
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(args.model).to(device)

state = torch.load(args.weights, map_location=device)
model.load_state_dict(state)
model.eval()


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
if args.model in ["inceptionv3"]:
    img_size = 299
else:
    img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------
def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(probs.argmax())
    pred_label = "DLBCL" if pred_class == 0 else "ENKTCL"

    return pred_label, probs[0], probs[1]


# ---------------------------------------------------------
# Run Inference
# ---------------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)
output_csv = os.path.join(args.output_dir, "patch_predictions.csv")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["patch", "prediction", "prob_DLBCL", "prob_ENKTCL"])

    for fname in sorted(os.listdir(args.input_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(args.input_dir, fname)
            label, p0, p1 = predict(path)

            writer.writerow([fname, label, f"{p0:.4f}", f"{p1:.4f}"])
            print(f"{fname} → {label} (DLBCL={p0:.3f}, ENKTCL={p1:.3f})")

print(f"\n✓ Inference completed.")
print(f"✓ Patch-level predictions saved to:\n  {output_csv}")
