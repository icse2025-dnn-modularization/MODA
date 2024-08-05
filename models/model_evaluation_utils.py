import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def _get_model_outputs(model, data_loader, device, num_classes=None, target_classes=None, show_progress=True):
    model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []
    pbar = tqdm(data_loader, desc="Collecting Model Outputs", disable=not show_progress)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if target_classes is not None:
            # use for evaluate module acc in case of it classifying only [target classes] instead of [all classes]
            assert num_classes
            if outputs.shape[1] != len(target_classes):
                outputs = outputs[:, target_classes]
            labels = F.one_hot(labels, num_classes=num_classes)[:, target_classes].argmax(dim=1)

        all_outputs.append(outputs)
        all_labels.append(labels)

    return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_model(model, data_loader, device, num_classes=None, target_classes=None,
                   acc_in_percent=False, show_progress=True):
    model.to(device)
    model.eval()

    outputs, labels = _get_model_outputs(model, data_loader, device,
                                         num_classes=num_classes,
                                         target_classes=target_classes,
                                         show_progress=show_progress)

    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    accuracy = correct / total
    if acc_in_percent:
        accuracy = accuracy * 100
    return accuracy


@torch.no_grad()
def evaluate_model_per_class(model, data_loader, device, num_classes, target_classes=None,
                             acc_in_percent=False, show_progress=True):
    model.to(device)
    model.eval()

    outputs, labels = _get_model_outputs(model, data_loader, device,
                                         num_classes=num_classes,
                                         target_classes=target_classes,
                                         show_progress=show_progress)

    _, predicted = torch.max(outputs, 1)
    mask_correct = (labels == predicted).unsqueeze(dim=1) * F.one_hot(labels, num_classes=num_classes)
    correct_per_class = mask_correct.sum(dim=0)

    total_per_class = labels.unique(return_counts=True)[1]

    accuracy_per_class = correct_per_class.float() / total_per_class.float()

    if acc_in_percent:
        accuracy_per_class *= 100

    return accuracy_per_class.tolist()


@torch.no_grad()
def evaluate_model_in_confusion_matrix(model, data_loader, device,
                                       num_classes=None, target_classes=None):
    model.to(device)
    model.eval()

    outputs, labels = _get_model_outputs(model, data_loader, device,
                                         num_classes=num_classes,
                                         target_classes=target_classes,
                                         show_progress=False)
    _, predicted = torch.max(outputs, 1)
    cm = confusion_matrix(labels.cpu(), predicted.cpu())
    return cm


def plot_confusion_matrix(cm, plot_title, annot_fmt=".2f"):
    sns.heatmap(cm, annot=True, cmap='vlag_r', fmt=annot_fmt, center=0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(plot_title)

    # highlight diagonal
    for i in range(len(cm)):
        plt.gca().add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2))
    plt.show()


@torch.no_grad()
def collect_model_output_minmax_per_class(model, data_loader, num_classes, device):
    model.to(device)
    model.eval()

    bound_values = [[float('inf'), float('-inf')], ] * num_classes  # [(min, max), ...]
    bound_values = torch.FloatTensor(bound_values).to(device)

    for images, labels in tqdm(data_loader, desc="Collecting Model Min-Max per Class"):
        images = images.to(device)
        outputs = model(images)

        for c in range(num_classes):
            class_outputs = outputs[:, c]
            bound_values[c][0] = min(bound_values[c][0], torch.min(class_outputs).item())
            bound_values[c][1] = max(bound_values[c][1], torch.max(class_outputs).item())

    return bound_values
