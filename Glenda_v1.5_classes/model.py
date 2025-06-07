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

# ----------------------------
# تنظیم device و seed
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --------------------------------------------------
# 0) تنظیمات اولیه
# --------------------------------------------------
BASE_DIR = Path('.')  # مسیر اصلی پروژه (فرض کنید اسکریپت در ریشه اجرا می‌شود)

# مسیرهای اصلیِ ورودی
FRAMES_DIR    = BASE_DIR / 'frames'          # تصاویر ورودی (.jpg)
ANNOTS_DIR    = BASE_DIR / 'annots'          # ماسک‌های رنگی (.png)
MASK_NUM_DIR  = BASE_DIR / 'masks_numeric'   # پوشه‌ی ماسک‌های عددی

# پوشه‌هایی که قرار است ساخته شوند:
TRAIN_FRAMES = BASE_DIR / 'train_frames'
TRAIN_MASKS  = BASE_DIR / 'train_masks'
VAL_FRAMES   = BASE_DIR / 'val_frames'
VAL_MASKS    = BASE_DIR / 'val_masks'
TEST_FRAMES  = BASE_DIR / 'test_frames'
TEST_MASKS   = BASE_DIR / 'test_masks'

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

# --------------------------------------------------
# 1) اگر ماسک‌های عددی ندارید، اول بسازید
# --------------------------------------------------
MASK_NUM_DIR.mkdir(exist_ok=True)
if not any(MASK_NUM_DIR.iterdir()):
    print("ساخت ماسک‌های عددی از پوشه‌ی annots/ ...")
    tol = 5
    count = 0
    for f in ANNOTS_DIR.glob('*.png'):
        arr = np.array(Image.open(f))
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        for idx, col in enumerate(CLASS_COLORS):
            mask[np.all(np.abs(arr - col) <= tol, axis=-1)] = idx
        out_path = MASK_NUM_DIR / f.name
        Image.fromarray(mask).save(out_path)
        count += 1
    print(f"تعداد {count} فایل ماسک عددی در {MASK_NUM_DIR} ذخیره شد.")
else:
    print("پوشه‌ی masks_numeric/ قبلاً پر شده. از آن استفاده خواهد شد.")

# --------------------------------------------------
# 2) لیست کردن همه‌ی جفت (تصویر, ماسک) برای Train+Val+Test
# --------------------------------------------------
all_frame_files = sorted(FRAMES_DIR.glob('*.jpg'))
all_mask_files  = sorted(MASK_NUM_DIR.glob('*.png'))

# مطمئن شوید که تعداد تصاویر و ماسک‌ها برابر است
assert len(all_frame_files) == len(all_mask_files), \
       f"تعداد تصاویر ({len(all_frame_files)}) با تعداد ماسک‌ها ({len(all_mask_files)}) برابر نیست!"

pairs = list(zip(all_frame_files, all_mask_files))
random.shuffle(pairs)

# --------------------------------------------------
# 3) تقسیم به سه زیرمجموعه: Train / Val / Test
# --------------------------------------------------
n_total = len(pairs)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val

train_pairs = pairs[:n_train]
val_pairs   = pairs[n_train : n_train + n_val]
test_pairs  = pairs[n_train + n_val :]

print(f"کل نمونه‌ها: {n_total}")
print(f"→ Train      : {len(train_pairs)}")
print(f"→ Validation : {len(val_pairs)}")
print(f"→ Test       : {len(test_pairs)}")

# --------------------------------------------------
# 4) ساخت پوشه‌های خروجی و کپی کردن فایل‌ها
# --------------------------------------------------
for d in [TRAIN_FRAMES, TRAIN_MASKS, VAL_FRAMES, VAL_MASKS, TEST_FRAMES, TEST_MASKS]:
    d.mkdir(exist_ok=True)

def copy_pair_list(pair_list, frame_dest, mask_dest):
    for (img_path, mask_path) in pair_list:
        dst_img = frame_dest / img_path.name
        shutil.copy(img_path, dst_img)
        dst_msk = mask_dest / mask_path.name
        shutil.copy(mask_path, dst_msk)

# 4.1) کپی Train
copy_pair_list(train_pairs, TRAIN_FRAMES, TRAIN_MASKS)
print(f"ファイル‌های Train در {TRAIN_FRAMES} و {TRAIN_MASKS} کپی شدند.")

# 4.2) کپی Validation
copy_pair_list(val_pairs, VAL_FRAMES, VAL_MASKS)
print(f"ファイル‌های Validation در {VAL_FRAMES} و {VAL_MASKS} کپی شدند.")

# 4.3) کپی Test
copy_pair_list(test_pairs, TEST_FRAMES, TEST_MASKS)
print(f"ファイル‌های Test در {TEST_FRAMES} و {TEST_MASKS} کپی شدند.")

# --------------------------------------------------
# 5) حالا متغیرهای train_loader و val_loader و test_loader را آماده کنید
# --------------------------------------------------
train_imgs, train_msks   = zip(*train_pairs)
val_imgs,   val_msks     = zip(*val_pairs)
test_imgs,  test_msks    = zip(*test_pairs)

# --------------------------------------------------
# 5.1) محاسبه وزن هر کلاس بر اساس فراوانی پیکسلی (برای Weighted CrossEntropy)
# **[CHANGE]**
# --------------------------------------------------
print("محاسبه فراوانی پیکسلی هر کلاس برای تولید class_weights ...")
pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
for m_path in train_msks:
    mask_np = np.array(Image.open(m_path))
    for cls in range(NUM_CLASSES):
        pixel_counts[cls] += int((mask_np == cls).sum())
total_pixels_all = pixel_counts.sum()
# وزن معکوس نسبت فراوانی: هرچه کلاس کمتر باشد، وزن بالاتری می‌گیرد
class_weights = total_pixels_all / (NUM_CLASSES * pixel_counts + 1e-6)
print(class_weights)
# نرمال‌سازی وزنه‌ها (اختیاری، تا مجموع به NUM_CLASSES برسد)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Pixel counts per class: {pixel_counts}")
print(f"Class weights: {class_weights}")

# --------------------------------------------------
# 5.2) Augmentationها (تقویت بیشتر)
# **[CHANGE]**
# --------------------------------------------------
# --------------------------------------------------
# Augmentationها (اصلاح‌شده)
# --------------------------------------------------
# --------------------------------------------------
# Augmentationها (اصلاح‌شده برای نسخه‌ی Albumentations شما)
# --------------------------------------------------
train_transform = A.Compose([
    # 1) ابتدا Resize به اندازه ثابت IMG_SIZE (512×512)
    A.Resize(*IMG_SIZE),

    # 2) سپس یک RandomResizedCrop با احتمال 0.5
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.7, 1.0), p=0.5),

    # 3) افکت‌های چرخشی و برگردان کردن تصادفی
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    # 4) تغییرات نور و رنگ
    A.RandomBrightnessContrast(p=0.3),

    # 5) تغییرات کششی نامنظم (Elastic) و بلور و اعوجاج شبکه‌ای
    A.ElasticTransform(p=0.2),
    A.GaussianBlur(p=0.2),
    A.GridDistortion(p=0.2),

    # 6) اعمال CoarseDropout (حذف تصادفی بخش‌های مربعی)
    A.CoarseDropout(
        max_holes=8,
        max_height=IMG_SIZE[0] // 10,   # حدوداً 51 پیکسل
        max_width=IMG_SIZE[1] // 10,    # حدوداً 51 پیکسل
        fill_value=0,
        p=0.3
    ),

    # 7) استانداردسازی (Normalize) با میانگین و انحراف معیار ImageNet
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),

    # 8) تبدیل نهایی به Tensor برای PyTorch
    ToTensorV2()
])

# --------------------------------------------------
# برای دقت: تعریف transform برای Validation و Test (بدون تغییر)
# --------------------------------------------------
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

# --------------------------------------------------
# 5.4) تعریف WeightedRandomSampler بهینه
# **[CHANGE]**
# --------------------------------------------------
def compute_sampler_weights(mask_paths):
    weights = []
    total_px = IMG_SIZE[0] * IMG_SIZE[1]
    for m_path in mask_paths:
        mask_np = np.array(Image.open(m_path).resize(IMG_SIZE, Image.NEAREST))
        lesion_count = int(np.count_nonzero(mask_np > 0))
        ratio = lesion_count / total_px
        # هرچه نسبت ضایعه بیشتر باشد، وزن sample بیشتر (مثلاً ضریب 10)
        weights.append(1.0 + ratio * 10.0)
    return weights

train_weights = compute_sampler_weights(train_msks)
train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

# --------------------------------------------------
# 5.5) ساخت DataLoaderها
# --------------------------------------------------
train_ds       = SegDataset(list(train_imgs), list(train_msks), transform=train_transform)
train_loader   = DataLoader(train_ds, batch_size=4, sampler=train_sampler, num_workers=4)

val_ds         = SegDataset(list(val_imgs), list(val_msks), transform=val_transform)
val_loader     = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

if len(test_pairs) > 0:
    test_ds     = SegDataset(list(test_imgs), list(test_msks), transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)
else:
    test_loader = None

print("\nSetup complete! You now have:")
print(f"  - {len(train_pairs)} training samples in {TRAIN_FRAMES}, {TRAIN_MASKS}")
print(f"  - {len(val_pairs)} validation samples in {VAL_FRAMES}, {VAL_MASKS}")
print(f"  - {len(test_pairs)}  testing samples in {TEST_FRAMES},  {TEST_MASKS}")

# --------------------------------------------------
# 6) تعریف مدل: UNet w/ ResNet34 encoder + ASPP + Attention
# (بدون تغییر نسبت به قبل)
# --------------------------------------------------
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

# --------------------------------------------------
# 7) Loss functions و optimizer
# --------------------------------------------------

# ===== Focal Tversky Loss =====
# می‌توانید به جای CompoundLoss یا CrossEntropyWeighted از این استفاده کنید.
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

# ===== CompoundLoss (CrossEntropy + Dice) =====
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

# ===== انتخاب Loss =====
use_focal_tversky = False   # اگر می‌خواهید از Focal Tversky استفاده کنید True کنید
use_weighted_ce = True      # اگر می‌خواهید از Weighted CrossEntropy در CompoundLoss استفاده کنید True کنید

if use_focal_tversky:
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75).to(device)
else:
    criterion = CompoundLoss(ce_weight=0.1, use_weighted_ce=use_weighted_ce).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

# --------------------------------------------------
# 8) Dice metric (unchanged)
# --------------------------------------------------
def dice_metric(logits, tgt, smooth=1e-6):
    preds = logits.argmax(1)                                           # [B, H, W]
    oh_t  = F.one_hot(tgt, NUM_CLASSES).permute(0,3,1,2).float()        # [B, C, H, W]
    oh_p  = F.one_hot(preds, NUM_CLASSES).permute(0,3,1,2).float()      # [B, C, H, W]
    inter = (oh_p * oh_t).sum(dim=[1,2,3])                              # [B]
    union = oh_p.sum([1,2,3]) + oh_t.sum([1,2,3])                       # [B]
    dices = (2*inter + smooth) / (union + smooth)                       # [B]
    return dices.mean().item()

# --------------------------------------------------
# 9) Training loop with early stopping
# --------------------------------------------------
history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
best_val = float('inf')
patience = 0
max_epochs = 100
warmup_epochs = 5

for ep in range(1, max_epochs + 1):
    # -- Training --
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {ep} [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # -- Validation --
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item()
            val_dice += dice_metric(out, y)
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)

    # Step scheduler on val_loss
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)

    print(f"Epoch {ep}: Train={train_loss:.4f}, Val={val_loss:.4f}, Dice={val_dice:.4f}")

    # Early stopping logic
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'best.pth')
        patience = 0
    else:
        patience += 1
        if patience >= 10:
            print("Early stopping triggered.")
            break

# --------------------------------------------------
# 10) Plot و ذخیره منحنی‌ها
# **[CHANGE]**: حالا منحنی‌ها را ذخیره کرده و نمایش می‌دهیم
# --------------------------------------------------
plt.figure()
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'],   label='Val Loss')
plt.legend()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')


plt.figure()
plt.plot(history['val_dice'], label='Val Dice')
plt.legend()
plt.savefig('val_dice_curve.png', dpi=300, bbox_inches='tight')


# --------------------------------------------------
# 11) ارزیابی روی Test Set (در صورت وجود)
# --------------------------------------------------
if test_loader is not None:
    print("\n======== Running on Test Set ========")
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss += criterion(out, y).item()
            test_dice += dice_metric(out, y)
    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    print(f"Test Loss={test_loss:.4f}, Test Dice={test_dice:.4f}")
else:
    print("\nNo test set was found, skipping final test evaluation.")

# --------------------------------------------------
# 12) کانورت ماسک عددی به رنگی و تابع Overlay
# --------------------------------------------------
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

# --------------------------------------------------
# 13) Overlay visualization on Validation set
# --------------------------------------------------
model.load_state_dict(torch.load('best.pth'))

output_dir = "val_visualizations"
os.makedirs(output_dir, exist_ok=True)

img_counter = 0
for x, y in val_loader:
    imgs = (x.cpu().permute(0,2,3,1).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    imgs = np.clip(imgs, 0, 1)
    imgs = (imgs * 255).astype(np.uint8)
    preds = model(x.to(device)).argmax(1).cpu().numpy()
    y_np   = y.cpu().numpy()

    for i in range(len(imgs)):
        img_rgb   = imgs[i]
        gt_mask   = color_mask(y_np[i])
        pred_mask = color_mask(preds[i])

        fig = plt.figure(figsize=(16,4))

        ax1 = fig.add_subplot(1,4,1)
        ax1.imshow(img_rgb); ax1.set_title("Input");           ax1.axis('off')

        ax2 = fig.add_subplot(1,4,2)
        ax2.imshow(gt_mask); ax2.set_title("Ground Truth");     ax2.axis('off')

        ax3 = fig.add_subplot(1,4,3)
        ax3.imshow(pred_mask); ax3.set_title("Prediction");     ax3.axis('off')

        overlay_img = overlay(img_rgb, pred_mask, alpha=0.5)
        ax4 = fig.add_subplot(1,4,4)
        ax4.imshow(overlay_img); ax4.set_title("Overlay");      ax4.axis('off')

        save_path = os.path.join(output_dir, f"viz_{img_counter:03d}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

        overlay_path = os.path.join(output_dir, f"overlay_trans_{img_counter:03d}.png")
        save_transparent_overlay(img_rgb, pred_mask, overlay_path, alpha=0.4)
        print(f"Saved transparent overlay {overlay_path}")

        img_counter += 1

    # اگر فقط یک batch می‌خواهید، خط زیر را uncomment کنید:
    # break

print("\nAll done! Train/Val/Test splits and final overlays have been saved.")

# --------------------------------------------------
# 14) تست نهایی (Test-set evaluation with metrics)
# --------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

if test_loader is not None:
    test_image_dir = Path("test_frames")
    test_mask_dir  = Path("test_masks")

    if test_image_dir.exists() and test_mask_dir.exists():
        test_imgs = sorted(test_image_dir.glob('*.jpg'))
        test_msks = sorted(test_mask_dir.glob('*.png'))

        if len(test_imgs) == len(test_msks) and len(test_imgs) > 0:
            print(f"\n✅ Found {len(test_imgs)} test images and masks. Starting test evaluation...")

            # Dataset و DataLoader قبلاً تعریف شدند (test_loader)
            model.load_state_dict(torch.load("best.pth"))
            model.eval()

            y_true_all, y_pred_all = [], []
            img_counter = 0
            output_dir = "test_visualizations"
            os.makedirs(output_dir, exist_ok=True)

            print("\n🧪 Running inference on test set...")
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc="Testing"):
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    preds = out.argmax(1)

                    # Denormalize برای نشان دادن تصویر
                    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
                    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
                    x_denorm = (x * std + mean).clamp(0,1)
                    imgs = (x_denorm.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

                    preds_np  = preds.cpu().numpy()
                    y_np      = y.cpu().numpy()

                    y_true_all.append(y_np.flatten())
                    y_pred_all.append(preds_np.flatten())

                    for i in range(len(imgs)):
                        img_rgb = imgs[i]
                        pred_mask = color_mask(preds_np[i])
                        overlay_img = overlay(img_rgb, pred_mask)

                        fig = plt.figure(figsize=(12, 4))
                        for j, (title, img) in enumerate([
                            ("Input", img_rgb),
                            ("Prediction", pred_mask),
                            ("Overlay", overlay_img)
                        ]):
                            ax = fig.add_subplot(1, 3, j+1)
                            ax.imshow(img)
                            ax.set_title(title)
                            ax.axis('off')

                        save_path = os.path.join(output_dir, f"test_viz_{img_counter:03d}.png")
                        fig.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"📸 Saved: {save_path}")
                        img_counter += 1

            # =========================================
            # محاسبه‌ی معیارها
            # =========================================
            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)

            mask_fg   = y_true_all > 0
            y_true_fg = y_true_all[mask_fg]
            y_pred_fg = y_pred_all[mask_fg]

            pixel_acc = accuracy_score(y_true_all, y_pred_all)
            acc_fg    = accuracy_score(y_true_fg, y_pred_fg)
            precision = precision_score(y_true_fg, y_pred_fg, average='macro', zero_division=0)
            recall    = recall_score(y_true_fg, y_pred_fg, average='macro', zero_division=0)
            f1        = f1_score(y_true_fg, y_pred_fg, average='macro', zero_division=0)
            iou       = jaccard_score(y_true_fg, y_pred_fg, average='macro', zero_division=0)
            dice      = 2 * (precision * recall) / (precision + recall + 1e-6)

            TP = np.logical_and(y_pred_all > 0, y_true_all > 0).sum()
            TN = np.logical_and(y_pred_all == 0, y_true_all == 0).sum()
            FP = np.logical_and(y_pred_all > 0, y_true_all == 0).sum()
            FN = np.logical_and(y_pred_all == 0, y_true_all > 0).sum()

            sensitivity  = TP / (TP + FN + 1e-6)
            specificity  = TN / (TN + FP + 1e-6)
            balanced_acc = (sensitivity + specificity) / 2

            print("\n📊 Test Evaluation Metrics:")
            print(f"Pixel Accuracy (overall): {pixel_acc:.4f}")
            print(f"Pixel Accuracy (lesions): {acc_fg:.4f}")
            print(f"Precision (macro):        {precision:.4f}")
            print(f"Recall (macro):           {recall:.4f}")
            print(f"F1-score (macro):         {f1:.4f}")
            print(f"Dice (macro):             {dice:.4f}")
            print(f"IoU (macro):              {iou:.4f}")
            print(f"Sensitivity:              {sensitivity:.4f}")
            print(f"Specificity:              {specificity:.4f}")
            print(f"Balanced Accuracy:        {balanced_acc:.4f}")
        else:
            print("⚠️ تعداد فایل‌های تست برابر نیست یا پوشه‌ها خالی هستند.")
    else:
        print("❌ پوشه‌های test_frames یا test_masks یافت نشد.")
