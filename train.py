"""Training entry point for LLFDisc.

The network and the objective are both resolved from the config file, so the
same script covers every experiment in the paper::

    python train.py --config configs/lol_v1.yaml

Checkpoints, the best-metric weights and a copy of the resolved config are
written under ``weight_save_path``. Training resumes automatically when
``checkpoint_path`` points at an existing checkpoint.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow running the script directly from the repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloaders
from utils import utils


def validate(model, dataloader, device, lpips_fn):
    """Evaluates the model on a paired dataloader.

    Args:
        model: Network to evaluate; switched to eval mode.
        dataloader: Paired loader yielding ``(low, high)`` batches.
        device: Device to run on.
        lpips_fn: Initialised ``lpips.LPIPS`` module.

    Returns:
        Tuple ``(avg_psnr, avg_ssim, avg_lpips)`` averaged over the loader.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            total_psnr += utils.calculate_psnr(output, high)
            total_ssim += utils.calculate_ssim(output, high)
            total_lpips += utils.calculate_lpips(output, high, lpips_fn)

    n = len(dataloader)
    return total_psnr / n, total_ssim / n, total_lpips / n


def _remove_stale_weights(saved_weight_files, tag):
    """Deletes previously saved weights whose name contains ``tag``.

    Only the current best file per metric is kept on disk, which matters because
    a long run can otherwise improve its best metric hundreds of times.

    Args:
        saved_weight_files: List of tracked paths; mutated in place.
        tag: Substring identifying the metric, e.g. ``"best_psnr"``.
    """
    for file_path in list(saved_weight_files):
        if os.path.exists(file_path) and tag in file_path:
            try:
                os.remove(file_path)
                print(f"Removed previous {tag} weights: {file_path}")
                saved_weight_files.remove(file_path)
            except OSError as e:
                print(f"Failed to remove {file_path}: {e}")


def _save_best(model, weight_save_path, saved_weight_files, tag, value, epoch):
    """Saves ``model``'s ``state_dict`` under a metric-tagged file name.

    Args:
        model: Network whose weights are saved.
        weight_save_path: Destination directory.
        saved_weight_files: List of tracked paths; the new path is appended.
        tag: Metric tag, e.g. ``"best_psnr"``.
        value: Metric value, embedded in the file name with the dot stripped.
        epoch: Epoch index, embedded in the file name.

    Returns:
        The path the weights were written to.
    """
    value_str = f"{tag}_{value:.4f}".replace(".", "")
    save_path = os.path.join(weight_save_path, f"{value_str}_e{epoch}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    saved_weight_files.append(save_path)
    return save_path


def main():
    import lpips

    parser = argparse.ArgumentParser(description='Train the LLFDisc model')
    parser.add_argument('--config', type=str, default='configs/lol_v1.yaml',
                        help='Path to the YAML config file (default: configs/lol_v1.yaml)')
    args = parser.parse_args()
    opt = utils.parse_config(args.config)

    # --- Hyperparameters and paths ---
    train_low = opt["train_low"]
    train_high = opt["train_high"]
    test_low = opt["test_low"]
    test_high = opt["test_high"]
    learning_rate = float(opt["learning_rate"])
    min_learning_rate = float(opt["min_learning_rate"])
    num_epochs = int(opt["num_epochs"])
    log_epochs = int(opt["log_epochs"])
    log_name = opt["log_name"]
    write_log = bool(opt["write_log"])

    model_path = opt["model_path"]
    class_name = opt["class_name"]
    loss_path = opt["loss_path"]
    loss_name = opt["loss_name"]

    device = torch.device(opt['device'] if torch.cuda.is_available() else 'cpu')
    batch_size = int(opt["batch_size"])
    weight_path = opt['weight_path']
    weight_save_path = opt['weight_save_path']
    checkpoint_save_path = opt['checkpoint_save_path']
    checkpoint_path = opt['checkpoint_path']
    checkpoint_epoch = int(opt['checkpoint_epoch'])
    crop_size = int(opt['crop_size'])
    patience = int(opt['patience'])
    seed = int(opt['seed'])
    # Dataloader worker processes. Lower this (0 disables multiprocessing) on
    # machines with few CPU cores or little RAM.
    num_workers = int(opt.get('num_workers', 4))
    curr_time = utils.get_current_time()

    alphas = {f'alpha{i}': float(opt[f'alpha{i}']) for i in range(1, 9)}
    alpha_lpips = float(opt.get('alpha_lpips', 0.1))

    os.makedirs(weight_save_path, exist_ok=True)
    os.makedirs(checkpoint_save_path, exist_ok=True)
    if write_log:
        os.makedirs('log', exist_ok=True)

    # Keep a copy of the config next to the weights for reproducibility.
    utils.save_config(opt, os.path.join(weight_save_path, "config.yaml"))

    # Fix every RNG so runs are reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    print("#######################")
    print("#### config below #####")
    print(f'Device: {device}')
    print(f'Batch_size: {batch_size}')
    print(f'LR: {learning_rate}, min_learning_rate: {min_learning_rate}')
    print(f'TotalEpochs: {num_epochs}, log_epochs: {log_epochs}')
    print(f'File_path_exists: {utils.check_file_exists(weight_path)}')
    print(f'Seed: {seed}')
    print(f'Model_path: {model_path}, class_name: {class_name}')
    print(f'patience: {patience}')
    print(f'checkpoint_epoch: {checkpoint_epoch}')
    print(f'checkpoint_save_path: {checkpoint_save_path}')
    print(f'checkpoint_path: {checkpoint_path}')
    print(f'loss_path: {loss_path}, loss_name: {loss_name}')
    print(f"alpha1:{alphas['alpha1']}, alpha2:{alphas['alpha2']}, "
          f"alpha3:{alphas['alpha3']}, alpha4:{alphas['alpha4']}")
    print(f"alpha5:{alphas['alpha5']}, alpha6:{alphas['alpha6']}, "
          f"alpha7:{alphas['alpha7']}, alpha8:{alphas['alpha8']}")
    print(f'alpha_lpips: {alpha_lpips}')
    print("#######################")

    # --- Data ---
    train_loader, test_loader = create_dataloaders(
        train_low, train_high, test_low, test_high,
        crop_size=crop_size, batch_size=batch_size, num_workers=num_workers)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # --- Model, loss, optimiser, scheduler ---
    network_cls = utils.dynamic_import(model_path, class_name)
    loss_cls = utils.dynamic_import(loss_path, loss_name)
    model = network_cls().to(device)

    criterion = loss_cls(device, opt)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = float('inf')  # LPIPS is a distance, so lower is better.
    start_epoch = 0

    counter = 0          # Consecutive epochs without an improvement.
    early_stop = False

    # Paths of every weight file written, so stale ones can be pruned.
    saved_weight_files = []

    # --- Resume from checkpoint ---
    if utils.check_file_exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Older checkpoints stored the bare state_dict.
            model.load_state_dict(checkpoint)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")

        if 'best_psnr' in checkpoint:
            best_psnr = checkpoint['best_psnr']
            print(f"Previous best PSNR: {best_psnr:.6f}")

        if 'best_ssim' in checkpoint:
            best_ssim = checkpoint['best_ssim']
            print(f"Previous best SSIM: {best_ssim:.6f}")

        if 'best_lpips' in checkpoint:
            best_lpips = checkpoint['best_lpips']
            print(f"Previous best LPIPS: {best_lpips:.6f}")

        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        if 'saved_weight_files' in checkpoint:
            saved_weight_files = checkpoint['saved_weight_files']

        # Sanity-check that the restored weights perform as recorded.
        avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, lpips_fn)
        print("#######################")
        print(f"Current model PSNR: {avg_psnr:.6f} (best recorded: {best_psnr:.6f})")
        print(f"Current model SSIM: {avg_ssim:.6f} (best recorded: {best_ssim:.6f})")
        print(f"Current model LPIPS: {avg_lpips:.6f} (best recorded: {best_lpips:.6f})")
        print("#######################")

    print('Training started.')

    log_file = f'log/{log_name}_{curr_time}.txt'
    if write_log:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f'opt:{opt} \n')

    start_time = time.time()
    total_avg_psnr, total_avg_ssim, total_avg_lpips = 0.0, 0.0, 0.0
    epoch = start_epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        current_lr = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # LPIPS is added on top of the configured objective; it expects
            # inputs in [-1, 1] rather than [0, 1].
            lpips_loss = lpips_fn(outputs * 2.0 - 1.0, targets * 2.0 - 1.0).mean()
            loss_total = loss + alpha_lpips * lpips_loss

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            current_lr = optimizer.param_groups[0]['lr']
            train_loss += loss_total.item()

        elapsed_epochs = epoch - start_epoch + 1

        if (epoch % log_epochs == 0) and write_log:
            train_loss = train_loss / len(train_loader)
            avg_loss = train_loss / elapsed_epochs
            every_epoch_psnr = total_avg_psnr / elapsed_epochs
            every_epoch_ssim = total_avg_ssim / elapsed_epochs
            every_epoch_lpips = total_avg_lpips / elapsed_epochs
            print(f'epoch:{epoch}, avg_loss:{avg_loss}')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f'Epoch:{epoch}, avg_loss:{avg_loss:.6f} | '
                    f'Every_epoch_psnr:{every_epoch_psnr:.6f} | '
                    f'every_epoch_ssim:{every_epoch_ssim:.6f} | '
                    f'every_epoch_lpips:{every_epoch_lpips:.6f} | '
                    f'current_lr:{current_lr}  \n')

        avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, lpips_fn)
        epoch_avg_loss = train_loss / (len(train_loader) * elapsed_epochs)
        total_avg_psnr += avg_psnr
        total_avg_ssim += avg_ssim
        total_avg_lpips += avg_lpips
        criterion.print_loss_info(epoch)
        print(f'Epoch {epoch + 1}/{num_epochs} | PSNR: {avg_psnr:.6f} | SSIM: {avg_ssim:.6f} | '
              f'LPIPS: {avg_lpips:.6f} | loss={epoch_avg_loss:.6f} | current_lr:{current_lr:.6f}')
        scheduler.step()

        # Full training state, so a run can be resumed exactly.
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_lpips': best_lpips,
            'scaler': scaler.state_dict(),
            'saved_weight_files': saved_weight_files,
        }

        if epoch % checkpoint_epoch == 0 and epoch > 0:
            periodic_checkpoint_path = os.path.join(
                checkpoint_save_path, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, periodic_checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch}")

            if write_log:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'Saved periodic checkpoint at epoch {epoch}\n')

        # --- Track the best model per metric ---
        improved = False

        if avg_psnr > best_psnr:
            _remove_stale_weights(saved_weight_files, "best_psnr")
            best_psnr = avg_psnr
            improved = True
            if write_log:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'## Epoch: {epoch}/{num_epochs}, Best_psnr:{best_psnr} \n')
            _save_best(model, weight_save_path, saved_weight_files, "best_psnr", best_psnr, epoch)
            print(f'Saving best PSNR model with PSNR: {best_psnr:.6f}')

        if avg_ssim > best_ssim:
            _remove_stale_weights(saved_weight_files, "best_ssim")
            best_ssim = avg_ssim
            improved = True
            if write_log:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'## Epoch: {epoch}/{num_epochs}, Best_ssim:{best_ssim} \n')
            _save_best(model, weight_save_path, saved_weight_files, "best_ssim", best_ssim, epoch)
            print(f'Saving best SSIM model with SSIM: {best_ssim:.6f}')

        if avg_lpips < best_lpips:
            _remove_stale_weights(saved_weight_files, "best_lpips")
            best_lpips = avg_lpips
            if write_log:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'## Epoch: {epoch}/{num_epochs}, Best_lpips:{best_lpips} \n')
            _save_best(model, weight_save_path, saved_weight_files, "best_lpips", best_lpips, epoch)
            print(f'Saving best LPIPS model with LPIPS: {best_lpips:.6f}')

        # Early stopping is driven by PSNR and SSIM only; LPIPS is reported but
        # does not reset the patience counter.
        if improved:
            counter = 0
        else:
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs! '
                      f'Best PSNR: {best_psnr:.6f}, Best SSIM: {best_ssim:.6f}')
                early_stop = True
                if write_log:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f'## Early stopping at Epoch: {epoch}/{num_epochs}, '
                                f'Best_psnr:{best_psnr}, Best_ssim:{best_ssim} \n')
                break

        if write_log and (epoch % log_epochs == 0):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'EarlyStopping counter: {counter} out of {patience}\n')

    total_training_time_minutes = (time.time() - start_time) / 60
    print(f'Total training time: {total_training_time_minutes:.2f} minutes')

    if early_stop:
        print(f'Training stopped early: no PSNR/SSIM improvement for {patience} epochs.')
    else:
        print(f'Training completed all {num_epochs} epochs.')

    # --- Clean up: keep only the best-PSNR, best-SSIM, best-LPIPS and final weights ---
    print("Pruning weight files, keeping only the best PSNR/SSIM/LPIPS models...")

    best_files = set()
    for tag, value in (("best_psnr", best_psnr), ("best_ssim", best_ssim), ("best_lpips", best_lpips)):
        marker = f"{tag}_{value:.4f}".replace(".", "")
        for file_path in saved_weight_files:
            if os.path.exists(file_path) and marker in file_path:
                best_files.add(file_path)

    final_epoch = epoch if early_stop else num_epochs - 1
    final_model_path = os.path.join(weight_save_path, f"final_epoch_{final_epoch}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final-epoch model: {final_model_path}")

    for file_path in saved_weight_files:
        if os.path.exists(file_path) and file_path not in best_files:
            try:
                os.remove(file_path)
                print(f"Removed non-optimal weights: {file_path}")
            except OSError as e:
                print(f"Failed to remove {file_path}: {e}")

    for file_path in sorted(best_files):
        print(f"Kept: {file_path}")
    print(f"Kept final-epoch model: {final_model_path}")


if __name__ == '__main__':
    main()
