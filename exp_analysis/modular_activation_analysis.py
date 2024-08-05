import os

import torch
from tqdm import tqdm
from configs import ModelConfig, BaseConfig
from dataset_loader import load_dataset, supported_std_datasets
from models import create_modular_model, compose_model_from_module_masks, supported_models
from models.model_utils import print_model_summary
from models.modular_utils import calculate_module_losses, mean_list


@torch.no_grad()
def measure_modular_metrics(model, train_loader):
    model.to(DEVICE)
    model.eval()

    pbar = tqdm(train_loader, desc="Inference")
    all_loss_overall, all_loss_ce, all_loss_affinity, \
        all_loss_dispersion, all_loss_compactness = [], [], [], [], []

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        modular_activations = model.get_modular_activations_of_current_batch()
        curr_loss_affinity, curr_loss_dispersion, \
            curr_loss_compactness = calculate_module_losses(modular_activations, labels)

        all_loss_affinity.append(curr_loss_affinity.detach())
        all_loss_dispersion.append(curr_loss_dispersion.detach())
        all_loss_compactness.append(curr_loss_compactness.detach())

    print({"loss_affinity": mean_list(all_loss_affinity),
           "loss_dispersion": mean_list(all_loss_dispersion),
           "loss_compactness": mean_list(all_loss_compactness)})


def main():
    num_classes, train_loader, test_loader = load_dataset(dataset_type=dataset_name, batch_size=128, num_workers=2)
    model = create_modular_model(model_type=model_name, num_classes=num_classes, modular_training_mode=True)

    # model.load_state_dict(torch.load(mt_model_save_path, map_location=DEVICE))
    model_checkpoint_dir = f"./temp/{model_name}_{dataset_name}/"
    model.load_pretrained_weights(
        os.path.join(model_checkpoint_dir, "model__bs128__ep200__lr0.05__aff1.0_dis1.0_comp0.3/model.pt"))
    print_model_summary(model)

    measure_modular_metrics(model, train_loader)


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    # model_name = "vgg16"
    # model_name = "resnet18"
    model_name = "mobilenet"

    # dataset_name = "svhn"
    # dataset_name = "cifar10"
    dataset_name = "cifar100"

    print(model_name, dataset_name)
    main()
