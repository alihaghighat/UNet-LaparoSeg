import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms, models
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# --------------------------------------------------
# 0) تنظیمات اولیه
# --------------------------------------------------
BASE_DIR = Path('.')  # مسیر اصلی پروژه (فرض کنید اسکریپت در ریشه اجرا می‌شود)


# نسبت تقسیم: 80% Train، 10% Val، 10% Test
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# Class color definitions (must match annotations)
CLASS_COLORS = [
    (0, 0, 0),        # background
    (190, 62, 204),   # Endo-Peritoneum
    (93, 89, 254),    # Endo-Ovar
    (145, 210, 138),  # Endo-TIE
    (238, 236, 50)    # Endo-Uterus
]
NUM_CLASSES = len(CLASS_COLORS)
IMG_SIZE    = (512, 512)
val_transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --------------------------------------------------
# 5.3) تعریف دیتاست و اعمال Augmentation
# **[CHANGE]**
# --------------------------------------------------
class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # خواندن تصویر و ماسک
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]))

        # اگر transform تعریف شده باشد، آن را به هردو تصویر و ماسک اعمال کن
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask  = transformed["mask"]
        else:
            # حداقل Resize و ToTensor
            image = transforms.ToTensor()(Image.fromarray(image).resize(IMG_SIZE))
            mask = torch.from_numpy(np.array(Image.fromarray(mask).resize(IMG_SIZE, Image.NEAREST))).long()

        return image, mask.long()

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.conv_1x1_output(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(self.relu(g1 + x1)))
        return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(out_ch, skip_ch, out_ch // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_ch + out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetResNet34Attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.init = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.e1 = base.layer1
        self.e2 = base.layer2
        self.e3 = base.layer3
        self.e4 = base.layer4
        self.aspp = ASPP(512, 256)
        self.d4 = DecoderBlock(256, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d2 = DecoderBlock(128, 64, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x0 = self.init(x)
        x1 = self.pool(x0)
        s1 = self.e1(x1)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.aspp(s4)
        d4 = self.d4(b, s3)
        d3 = self.d3(d4, s2)
        d2 = self.d2(d3, s1)
        d1 = self.d1(d2, x0)
        return F.interpolate(self.final(d1), size=x.shape[2:], mode='bilinear', align_corners=False)

model = UNetResNet34Attention(num_classes=NUM_CLASSES).to(device)
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets):
        # preds: [B, C, H, W], logits
        preds = F.softmax(preds, dim=1)
        one_hot = F.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float()
        TP = (preds * one_hot).sum(dim=(2, 3))
        FP = (preds * (1 - one_hot)).sum(dim=(2, 3))
        FN = ((1 - preds) * one_hot).sum(dim=(2, 3))
        tversky = (TP + 1e-6) / (TP + self.alpha * FP + self.beta * FN + 1e-6)
        loss = (1 - tversky).pow(self.gamma).mean()
        return loss
# تعریف class weights دستی
class_weights_tensor = torch.tensor(
    [0.0080999, 0.25035065, 0.43587076, 0.49093265, 3.81474603],
    dtype=torch.float32
).to(device)

class CompoundLoss(nn.Module):
    def __init__(self, ce_weight=0.2, use_weighted_ce=False):
        super().__init__()
        if use_weighted_ce:
            # **[CHANGE]**: Weighted CrossEntropy با class_weights
            self.ce = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.w  = ce_weight

    def forward(self, out, tgt):
        # out: [B, C, H, W], tgt: [B, H, W]
        ce_l = self.ce(out, tgt)
        ps   = F.softmax(out, 1)                                        # [B, C, H, W]
        toh  = F.one_hot(tgt, NUM_CLASSES).permute(0,3,1,2).float()      # [B, C, H, W]
        inter = (ps * toh).sum([0,2,3])                                  # [C]
        union = ps.sum([0,2,3]) + toh.sum([0,2,3])                       # [C]
        dice_l = 1 - ((2*inter + 1e-6) / (union + 1e-6)).mean()          # scalar
        return self.w * ce_l + (1 - self.w) * dice_l
use_focal_tversky = False   # اگر می‌خواهید از Focal Tversky استفاده کنید True کنید
use_weighted_ce = True      # اگر می‌خواهید از Weighted CrossEntropy در CompoundLoss استفاده کنید True کنید

if use_focal_tversky:
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75).to(device)
else:
    criterion = CompoundLoss(ce_weight=0.1, use_weighted_ce=use_weighted_ce).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
def color_mask(mask):
    h, w = mask.shape
    cm   = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        cm[mask == i] = c
    return cm

def overlay(image, mask_color, alpha=0.5):
    overlayed = image.copy()
    overlayed = (1 - alpha) * overlayed + alpha * mask_color
    return overlayed.astype(np.uint8)

from PIL import Image
def save_transparent_overlay(img_np, mask_np, out_path, alpha=0.5):
    pil_img  = Image.fromarray(img_np).convert("RGBA")
    pil_mask = Image.fromarray(mask_np).convert("RGBA")
    a = int(255 * alpha)
    alpha_channel = Image.new("L", pil_mask.size, a)
    pil_mask.putalpha(alpha_channel)
    comp = Image.alpha_composite(pil_img, pil_mask)
    comp.save(out_path, format="PNG")
model.load_state_dict(torch.load('best.pth'))
def predict_single_image(image_path, save_dir="output_single", alpha=0.5):
    os.makedirs(save_dir, exist_ok=True)

    # Load image
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)

    # Preprocess
    transform = A.Compose([
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    input_tensor = transform(image=original_np)["image"].unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).squeeze().cpu().numpy()

    # Colorize mask
    pred_color = color_mask(pred)

    # Resize original for overlay
    original_resized = np.array(original.resize(IMG_SIZE))
    overlayed = overlay(original_resized, pred_color, alpha=alpha)

    # Save images
    basename = Path(image_path).stem
    mask_path = f"{save_dir}/{basename}_mask.png"
    overlay_path = f"{save_dir}/{basename}_overlay.png"
    Image.fromarray(pred_color).save(mask_path)
    Image.fromarray(overlayed).save(overlay_path)

    print(f"✅ Saved mask and overlay to '{save_dir}'")

    # ----------- Generate Enhanced English Report -----------
    class_names = [
        "Background",
        "Endo-Peritoneum",
        "Endo-Ovary",
        "Endo-TIE (Tubal/Intestinal Endometriosis)",
        "Endo-Uterus"
    ]
    descriptions = {
        1: "Superficial peritoneal lesions suggestive of early-stage endometriosis were identified. These typically appear as small, pigmented patches on the peritoneal surface and may indicate subtle disease onset.",
        2: "Ovarian abnormalities consistent with endometriomas (chocolate cysts) or post-inflammatory adhesions were detected. These findings may correlate with chronic pelvic pain and infertility.",
        3: "Deeply infiltrating lesions were observed in the tubal or intestinal regions, suggesting advanced endometriotic involvement. These lesions may penetrate beyond the serosa and involve adjacent pelvic organs.",
        4: "Surface-level uterine abnormalities indicative of serosal endometriosis were found. This may reflect external infiltration without affecting the endometrium itself."
    }

    present_classes = np.unique(pred)
    report_lines = [f"🧾 Segmentation Summary Report for '{basename}'"]
    report_lines.append("The AI model examined the laparoscopic image and identified the following endometriosis-related features:")

    found = False
    for cls in present_classes:
        if cls == 0:
            continue  # skip background
        found = True
        name = class_names[cls]
        desc = descriptions.get(cls, "Description for this class is not available.")
        report_lines.append(f"\n🔹 **{name}**\n   - {desc}")

    if not found:
        report_lines.append("\n✅ No pathological signs of endometriosis were detected in this image.")

    report_lines.append("\n📌 *Note: This report is generated by an AI-based segmentation model and is intended to assist clinical interpretation. It should not be used as a substitute for professional medical judgment.*")

    full_report = "\n".join(report_lines)
    print(full_report)

    # Save report to text file
    with open(f"{save_dir}/{basename}_report.txt", "w") as f:
        f.write(full_report)

    # ----------- Show visualization -----------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_resized)
    plt.title("Input"); plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(pred_color)
    plt.title("Prediction"); plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(overlayed)
    plt.title("Overlay"); plt.axis("off")
    plt.tight_layout()
    plt.show()

    return full_report

predict_single_image("test_frames/c_1_v_(video_7.mp4)_f_1299.jpg")
