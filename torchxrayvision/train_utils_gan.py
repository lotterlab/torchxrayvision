"""
Training utilities for GAN-based windowing with two discriminators.
"""

import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def save_checkpoint(generator, disc_real_fake, disc_dataset, 
                   opt_gen, opt_disc_rf, opt_disc_ds,
                   epoch, cfg, filename='checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'disc_real_fake_state_dict': disc_real_fake.state_dict(),
        'disc_dataset_state_dict': disc_dataset.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_rf_state_dict': opt_disc_rf.state_dict(),
        'opt_disc_ds_state_dict': opt_disc_ds.state_dict(),
        'cfg': cfg
    }
    torch.save(checkpoint, os.path.join(cfg.output_dir, filename))
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(checkpoint_path, generator, disc_real_fake, disc_dataset,
                   opt_gen, opt_disc_rf, opt_disc_ds):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    disc_real_fake.load_state_dict(checkpoint['disc_real_fake_state_dict'])
    disc_dataset.load_state_dict(checkpoint['disc_dataset_state_dict'])
    opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
    opt_disc_rf.load_state_dict(checkpoint['opt_disc_rf_state_dict'])
    opt_disc_ds.load_state_dict(checkpoint['opt_disc_ds_state_dict'])
    return checkpoint['epoch']


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolate between real and fake
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def compute_regularization_loss(original_images, windowed_images, reg_type='l2'):
    """
    Compute regularization loss to preserve morphology.
    
    Args:
        original_images: (B, C, H, W) original images
        windowed_images: (B, C, H, W) windowed images
        reg_type: type of regularization ('l2', 'l1', 'perceptual')
    """
    if reg_type == 'l2':
        return torch.nn.functional.mse_loss(windowed_images, original_images)
    elif reg_type == 'l1':
        return torch.nn.functional.l1_loss(windowed_images, original_images)
    elif reg_type == 'perceptual':
        # Simple perceptual loss using pixel gradients
        # For more sophisticated version, use VGG features
        grad_x_orig = original_images[:, :, :, 1:] - original_images[:, :, :, :-1]
        grad_y_orig = original_images[:, :, 1:, :] - original_images[:, :, :-1, :]
        grad_x_wind = windowed_images[:, :, :, 1:] - windowed_images[:, :, :, :-1]
        grad_y_wind = windowed_images[:, :, 1:, :] - windowed_images[:, :, :-1, :]
        
        loss_x = torch.nn.functional.mse_loss(grad_x_wind, grad_x_orig)
        loss_y = torch.nn.functional.mse_loss(grad_y_wind, grad_y_orig)
        return loss_x + loss_y
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def save_sample_images(original, windowed, epoch, cfg, num_samples=8):
    """Save sample images for visualization."""
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    
    for i in range(min(num_samples, original.size(0))):
        # Original
        axes[0, i].imshow(original[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
            
        # Windowed
        axes[1, i].imshow(windowed[i, 0].cpu().detach().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Windowed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, f'samples_epoch_{epoch}.png'))
    plt.close()


def train_gan(generator, disc_real_fake, disc_dataset, train_dataset, 
              valid_dataset, dataset_labels, cfg):
    """
    Main training loop for GAN with two discriminators.
    
    Args:
        generator: Generator model (hypernetwork)
        disc_real_fake: Discriminator for real vs windowed
        disc_dataset: Discriminator for dataset source
        train_dataset: Training dataset
        valid_dataset: Validation dataset (optional)
        dataset_labels: Tensor of dataset labels for each sample
        cfg: Configuration object
    """
    device = torch.device('cuda' if cfg.cuda and torch.cuda.is_available() else 'cpu')
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.threads,
        pin_memory=True
    )
    
    # Loss functions
    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=cfg.lr_gen, 
                        betas=(cfg.beta1, cfg.beta2))
    opt_disc_rf = optim.Adam(disc_real_fake.parameters(), lr=cfg.lr_disc,
                             betas=(cfg.beta1, cfg.beta2))
    opt_disc_ds = optim.Adam(disc_dataset.parameters(), lr=cfg.lr_disc,
                             betas=(cfg.beta1, cfg.beta2))
    
    # Learning rate schedulers (optional)
    if cfg.use_scheduler if hasattr(cfg, 'use_scheduler') else False:
        scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=30, gamma=0.1)
        scheduler_disc_rf = optim.lr_scheduler.StepLR(opt_disc_rf, step_size=30, gamma=0.1)
        scheduler_disc_ds = optim.lr_scheduler.StepLR(opt_disc_ds, step_size=30, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if cfg.resume:
        start_epoch = load_checkpoint(cfg.resume, generator, disc_real_fake, 
                                     disc_dataset, opt_gen, opt_disc_rf, opt_disc_ds)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'epoch': [],
        'loss_gen': [],
        'loss_disc_rf': [],
        'loss_disc_ds': [],
        'loss_reg': [],
        'acc_disc_rf': [],
        'acc_disc_ds': []
    }
    
    # Labels for real/fake discriminator
    real_label = 1.0
    fake_label = 0.0
    
    print(f"Starting training for {cfg.num_epochs} epochs...")
    
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start_time = time.time()
        
        # Training metrics
        gen_losses = []
        disc_rf_losses = []
        disc_ds_losses = []
        reg_losses = []
        disc_rf_accs = []
        disc_ds_accs = []
        
        generator.train()
        disc_real_fake.train()
        disc_dataset.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['img'].to(device)
            batch_size = images.size(0)
            
            # Get dataset labels for this batch
            # Assuming dataset labels are aligned with dataset ordering
            batch_dataset_labels = dataset_labels[:batch_size].to(device)
            
            # =======================
            # Train Discriminators
            # =======================
            
            # Train Real/Fake Discriminator
            opt_disc_rf.zero_grad()
            
            # Real images
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            real_output = disc_real_fake(images)
            loss_disc_rf_real = criterion_bce(real_output, real_labels)
            
            # Fake (windowed) images
            with torch.no_grad():
                windowed_images, _ = generator(images)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)
            fake_output = disc_real_fake(windowed_images.detach())
            loss_disc_rf_fake = criterion_bce(fake_output, fake_labels)
            
            # Combined loss
            loss_disc_rf = (loss_disc_rf_real + loss_disc_rf_fake) / 2
            
            # Gradient penalty for WGAN-GP (optional)
            if cfg.use_wgan:
                gp = compute_gradient_penalty(disc_real_fake, images, 
                                             windowed_images.detach(), device)
                loss_disc_rf = loss_disc_rf + cfg.gp_lambda * gp
            
            loss_disc_rf.backward()
            opt_disc_rf.step()
            
            # Compute accuracy
            real_acc = ((real_output > 0.5).float() == real_labels).float().mean()
            fake_acc = ((fake_output < 0.5).float() == fake_labels).float().mean()
            disc_rf_acc = (real_acc + fake_acc) / 2
            
            # Train Dataset Discriminator
            opt_disc_ds.zero_grad()
            
            # ONLY train on windowed images (not original images)
            # This gives generator a fair chance to fool the discriminator
            ds_output_fake = disc_dataset(windowed_images.detach())
            loss_disc_ds = criterion_ce(ds_output_fake, batch_dataset_labels)
            
            loss_disc_ds.backward()
            opt_disc_ds.step()
            
            # Compute accuracy (only on windowed images now)
            _, predicted = torch.max(ds_output_fake, 1)
            disc_ds_acc = (predicted == batch_dataset_labels).float().mean()
            
            # =======================
            # Train Generator
            # =======================
            
            # Train generator every n_critic iterations
            if batch_idx % cfg.n_critic == 0:
                opt_gen.zero_grad()
                
                # Generate windowed images
                windowed_images, spline_params = generator(images)
                
                # Loss 1: Fool real/fake discriminator (want to be classified as real)
                gen_output_rf = disc_real_fake(windowed_images)
                loss_gen_rf = criterion_bce(gen_output_rf, real_labels)
                
                # Loss 2: Fool dataset discriminator 
                # Strategy: Train generator to predict WRONG dataset labels
                # This forces it to remove dataset-specific features
                ds_output = disc_dataset(windowed_images)
                
                # Create wrong labels (flip between datasets)
                num_datasets = ds_output.size(1)
                wrong_labels = (batch_dataset_labels + 1) % num_datasets
                
                # Generator wants to be classified as the WRONG dataset
                loss_gen_ds = criterion_ce(ds_output, wrong_labels)
                
                # Loss 3: Regularization to preserve morphology
                loss_reg = compute_regularization_loss(
                    images, windowed_images, reg_type=cfg.reg_type)
                
                # Combined generator loss
                loss_gen = (cfg.lambda_adv * loss_gen_rf + 
                           cfg.lambda_dataset * loss_gen_ds +
                           cfg.lambda_reg * loss_reg)
                
                loss_gen.backward()
                opt_gen.step()
                
                # Store losses
                gen_losses.append(loss_gen.item())
                reg_losses.append(loss_reg.item())
            
            disc_rf_losses.append(loss_disc_rf.item())
            disc_ds_losses.append(loss_disc_ds.item())
            disc_rf_accs.append(disc_rf_acc.item())
            disc_ds_accs.append(disc_ds_acc.item())
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{np.mean(gen_losses[-10:]) if gen_losses else 0:.3f}',
                'D_RF': f'{np.mean(disc_rf_losses[-10:]):.3f}',
                'D_DS': f'{np.mean(disc_ds_losses[-10:]):.3f}',
                'Acc_RF': f'{np.mean(disc_rf_accs[-10:]):.3f}',
                'Acc_DS': f'{np.mean(disc_ds_accs[-10:]):.3f}'
            })
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_gen_loss = np.mean(gen_losses) if gen_losses else 0
        avg_disc_rf_loss = np.mean(disc_rf_losses)
        avg_disc_ds_loss = np.mean(disc_ds_losses)
        avg_reg_loss = np.mean(reg_losses) if reg_losses else 0
        avg_disc_rf_acc = np.mean(disc_rf_accs)
        avg_disc_ds_acc = np.mean(disc_ds_accs)
        
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs} - Time: {epoch_time:.2f}s")
        print(f"  Gen Loss: {avg_gen_loss:.4f}")
        print(f"  Disc RF Loss: {avg_disc_rf_loss:.4f} | Acc: {avg_disc_rf_acc:.4f}")
        print(f"  Disc DS Loss: {avg_disc_ds_loss:.4f} | Acc: {avg_disc_ds_acc:.4f}")
        print(f"  Reg Loss: {avg_reg_loss:.4f}")
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['loss_gen'].append(avg_gen_loss)
        history['loss_disc_rf'].append(avg_disc_rf_loss)
        history['loss_disc_ds'].append(avg_disc_ds_loss)
        history['loss_reg'].append(avg_reg_loss)
        history['acc_disc_rf'].append(avg_disc_rf_acc)
        history['acc_disc_ds'].append(avg_disc_ds_acc)
        
        # Save sample images
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                sample_images = sample_batch['img'][:8].to(device)
                sample_windowed, _ = generator(sample_images)
                save_sample_images(sample_images, sample_windowed, epoch + 1, cfg)
        
        # Save checkpoint
        if (epoch + 1) % cfg.save_freq == 0:
            save_checkpoint(generator, disc_real_fake, disc_dataset,
                          opt_gen, opt_disc_rf, opt_disc_ds,
                          epoch + 1, cfg, 
                          filename=f'checkpoint_epoch_{epoch+1}.pth')
        
        # Update learning rate
        if hasattr(cfg, 'use_scheduler') and cfg.use_scheduler:
            scheduler_gen.step()
            scheduler_disc_rf.step()
            scheduler_disc_ds.step()
    
    # Save final model
    save_checkpoint(generator, disc_real_fake, disc_dataset,
                   opt_gen, opt_disc_rf, opt_disc_ds,
                   cfg.num_epochs, cfg, filename='final_model.pth')
    
    # Save training history
    np.save(os.path.join(cfg.output_dir, 'training_history.npy'), history)
    
    # Plot training curves
    plot_training_curves(history, cfg)
    
    print("Training completed!")
    return history


def plot_training_curves(history, cfg):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # Generator loss
    axes[0, 0].plot(epochs, history['loss_gen'])
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Discriminator RF loss
    axes[0, 1].plot(epochs, history['loss_disc_rf'])
    axes[0, 1].set_title('Discriminator (Real/Fake) Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Discriminator DS loss
    axes[0, 2].plot(epochs, history['loss_disc_ds'])
    axes[0, 2].set_title('Discriminator (Dataset) Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    
    # Regularization loss
    axes[1, 0].plot(epochs, history['loss_reg'])
    axes[1, 0].set_title('Regularization Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Discriminator RF accuracy
    axes[1, 1].plot(epochs, history['acc_disc_rf'])
    axes[1, 1].set_title('Discriminator (Real/Fake) Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True)
    
    # Discriminator DS accuracy
    axes[1, 2].plot(epochs, history['acc_disc_ds'])
    axes[1, 2].axhline(y=1.0/len(history['acc_disc_ds']), color='r', 
                      linestyle='--', label='Random')
    axes[1, 2].set_title('Discriminator (Dataset) Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'training_curves.png'))
    plt.close()