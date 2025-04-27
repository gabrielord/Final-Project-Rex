import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from captum.attr import IntegratedGradients
from lime import lime_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

def build_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def build_explainers(model):
    ig = IntegratedGradients(model)
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    explainer = lime_image.LimeImageExplainer()
    return ig, cam, explainer

def predict_fn(images, model, transform, device):
    model.eval()
    images = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
    images = images.to(device)
    with torch.no_grad():
        preds = model(images)
        probs = torch.softmax(preds, dim=1)
    return probs.detach().cpu().numpy()

def run_explanations(model, val_loader, ig, cam, explainer, transform, device, max_images=5):
    image_counter = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for i in range(inputs.size(0)):
            if image_counter >= max_images:
                return

            input_tensor = inputs[i].unsqueeze(0)

            # Denormalize image
            denorm_input = inputs[i].cpu().clone()
            denorm_input *= torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denorm_input += torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            denorm_input = denorm_input.clamp(0, 1)
            img = (denorm_input.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Grad-CAM
            grayscale_cam = cam(input_tensor, targets=[ClassifierOutputTarget(int(labels[i]))])[0]
            cam_image = show_cam_on_image(denorm_input.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

            # Integrated Gradients
            baseline = torch.zeros_like(input_tensor).to(device)
            attributions = ig.attribute(input_tensor, baseline, target=int(labels[i]), n_steps=50)
            attribution = attributions.squeeze().cpu().permute(1, 2, 0).detach().numpy()
            attribution = np.abs(attribution).mean(axis=2)
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
            ig_image = (attribution * 255).astype(np.uint8)

            # LIME
            explanation = explainer.explain_instance(
                img, lambda imgs: predict_fn(imgs, model, transform, device),
                top_labels=2, hide_color=0, num_samples=1000
            )
            temp, mask = explanation.get_image_and_mask(
                label=int(labels[i]),
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            lime_image_result = mark_boundaries(temp, mask)

            # Plot and save
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(cam_image)
            axs[0].set_title('Grad-CAM')
            axs[0].axis('off')

            axs[1].imshow(ig_image, cmap='hot')
            axs[1].set_title('Integrated Gradients')
            axs[1].axis('off')

            axs[2].imshow(lime_image_result)
            axs[2].set_title('LIME')
            axs[2].axis('off')

            plt.tight_layout()
            fig.savefig(f'comparative_outputs/comparison_{image_counter}_new.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            image_counter += 1

@torch.no_grad()          # gradients not needed for the masking loops
def _deletion_insertion_curve(
    model: torch.nn.Module,
    x: torch.Tensor,                    # 1 × C × H × W
    sal: np.ndarray,                    # H × W, normalised [0,1]
    cls: int,
    mode: str = "deletion",
    steps: int = 100
):
    """Return confidence curve (length ≤ steps)."""
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be 'deletion' or 'insertion'")

    device = x.device
    sal_flat = torch.from_numpy(sal).to(device).flatten()
    order = sal_flat.argsort(descending=True) 

    _, _, H, W = x.shape
    pix_per   = max(1, (H * W) // steps)

    # start image
    mod = x.clone() if mode == "deletion" else torch.zeros_like(x)

    scores = []
    for k in range(0, H * W, pix_per):
        prob = F.softmax(model(mod), 1)[0, cls].item()
        scores.append(prob)

        idx = order[k: k + pix_per]
        if idx.numel() == 0:
            break
        y, xcoord = idx // W, idx % W
        if mode == "deletion":
            mod[:, :, y, xcoord] = 0
        else: # insertion
            mod[:, :, y, xcoord] = x[:, :, y, xcoord]
    return scores


def _auc(curve, mode):
    x = np.linspace(0, 1, len(curve))
    y = np.array(curve)
    return metrics.auc(x, 1 - y) if mode == "deletion" else metrics.auc(x, y)


def evaluate_method(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    explainer_fn, # (tensor, cls) → saliency (H,W)
    name: str,
    device: torch.device,
    max_imgs: int = 30,
    steps: int = 100,
):
    model.eval()
    del_auc_list, ins_auc_list = [], []
    processed = 0

    for xb, yb in tqdm(loader, desc=f"Evaluating {name}"):
        xb, yb = xb.to(device), yb.to(device)
        for img, lbl in zip(xb, yb):
            if processed >= max_imgs:
                break
            img1 = img.unsqueeze(0)
            cls  = int(lbl)

            sal = explainer_fn(img1, cls)              # H × W
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            # sal = sal.astype(np.float32)

            d_curve = _deletion_insertion_curve(model, img1, sal, cls, "deletion",  steps)
            i_curve = _deletion_insertion_curve(model, img1, sal, cls, "insertion", steps)

            del_auc_list.append(_auc(d_curve, "deletion"))
            ins_auc_list.append(_auc(i_curve, "insertion"))

            processed += 1
        if processed >= max_imgs:
            break

    return float(np.mean(del_auc_list)), float(np.mean(ins_auc_list))