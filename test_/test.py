import os
import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
#todo

def denormalize_image_mean_std(image_array, mean, std, rmin,rmax):
    
    denormalized = (image_array * std) + mean  # Denormalizza con mean e std

    # Normalizzazione min-max per portare i valori tra 0 e 1
    normalized = (denormalized - rmin) / (rmax - rmin)
    
    return normalized

# Funzione per calcolare l'NDVI
def calculate_ndvi(nir_band, red_band):
    # Calcolo NDVI: (NIR - RED) / (NIR + RED)
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)  
    return ndvi

# Funzione per visualizzare e salvare l'NDVI
def display_ndvi_stretched(ndvi, title, index, output_dir):
    # Assicurati che l'array NDVI sia a singolo canale 
    ndvi = ndvi.squeeze().numpy()  

    # Visualizza l'NDVI con una colormap adeguata
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar()  # Mostra la barra dei colori
    plt.title(f"{title} - Immagine {index}")
    
    # Salva l'immagine su file
    save_path = os.path.join(output_dir, f'{title}_ndvi_{index}.png')
    plt.savefig(save_path, format='png')
    
    # Chiudi la figura per liberare memoria
    plt.close()

def test(generator, dataloader, device, output_dir, config):
    # Carica il modello dal checkpoint desiderato
    checkpoint_path = f"{config.save_dir}/generator_epoch_{config.num_epochs}.pth"
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    total_ssim = 0.0
    total_psnr = 0.0
    total_ssim_ndvi = 0.0
    total_psnr_ndvi = 0.0
    count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):

            red_img, green_img = sample['image_r'].to(device), sample['image_g'].to(device)
            target_img = sample['image_nir'].to(device)

            fake_img = generator(red_img, green_img)


            # Ciclo attraverso ogni immagine nel batch
            for j in range(dataloader.batch_size):
                
                red_img[j] = denormalize_image_mean_std(red_img[j], config.mean_r, config.std_r, config.min_r,config.max_r)
                green_img[j] = denormalize_image_mean_std(green_img[j], config.mean_g, config.std_g, config.min_g,config.max_g)
                fake_img[j] = denormalize_image_mean_std(fake_img[j], config.mean_n, config.std_n,config.min_n,config.max_n)
                target_img[j] = denormalize_image_mean_std(target_img[j], config.mean_n, config.std_n, config.min_n,config.max_n)
            



                # Salva red, green, fake, target images
                #save_image(red_img[j], os.path.join(output_dir, f'red_image_{i+1}_{j+1}.TIFF'), nrow=1)
                #save_image(green_img[j], os.path.join(output_dir, f'green_image_{i+1}_{j+1}.TIFF'), nrow=1)
                #save_image(fake_img[j], os.path.join(output_dir, f'fake_image_{i+1}_{j+1}.TIFF'), nrow=1)
                #save_image(target_img[j], os.path.join(output_dir, f'target_image_{i+1}_{j+1}.TIFF'), nrow=1)

                # Concatenare le immagini (adatta per ogni immagine del batch)
                combined = torch.cat([red_img[j], green_img[j], fake_img[j], target_img[j]], dim=2)

                save_image(combined, os.path.join(output_dir, f'fake_image_{i+1}_{j+1}.png'), nrow=1)
                
               

                # Calcola l'NDVI per le immagini generate (fake) e reali (target)
                ndvi_image_fake = calculate_ndvi(fake_img[j], red_img[j])
                ndvi_image_true = calculate_ndvi(target_img[j], red_img[j])
                
                # Visualizza e salva le immagini NDVI
                display_ndvi_stretched(ndvi_image_fake, "Fake", f"{i+1}_{j+1}", output_dir)
                display_ndvi_stretched(ndvi_image_true, "True", f"{i+1}_{j+1}", output_dir)

                #----------------------------------------------------------------

                # Calcola PSNR per le immagini
                fake_img_np = fake_img[j].numpy()
                target_img_np = target_img[j].numpy()

                psnr_value = psnr(fake_img_np, target_img_np)
                
                total_psnr += psnr_value.item()

                # Calcola SSIM per le immagini
                min_dim = min(target_img_np.shape[-2:])
                win_size = 3 if min_dim >= 7 else min_dim
                ssim_value, _ = ssim(target_img_np, fake_img_np, full=True, win_size=win_size, channel_axis=0, data_range=1.0)
                total_ssim += ssim_value
                #----------------------------------------------------------------

                # Calcola PSNR per NDVI
                ndvi_image_fake_np = ndvi_image_fake[j].numpy()
                ndvi_image_true_np = ndvi_image_true[j].numpy()
                psnr_value_ndvi = psnr(ndvi_image_fake_np, ndvi_image_true_np, data_range=2)
                total_psnr_ndvi += psnr_value_ndvi.item()

                # Calcola SSIM per NDVI
                ssim_value_ndvi, _ = ssim(ndvi_image_true_np, ndvi_image_fake_np, full=True, win_size=3, data_range=2)
                total_ssim_ndvi += ssim_value_ndvi
                #----------------------------------------------------------------
                count += 1
                print(f'psnr to {psnr_value_ndvi.item()}')
                print(f'ssim to {ssim_value_ndvi}')

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_psnr_ndvi = total_psnr_ndvi / count
    avg_ssim_ndvi = total_ssim_ndvi / count

    print(f'Test PSNR su foto: {avg_psnr:.4f}')
    print(f'Test SSIM su foto: {avg_ssim:.4f}')
    print(f'Test PSNR su NDVI: {avg_psnr_ndvi:.4f}')
    print(f'Test SSIM su NDVI: {avg_ssim_ndvi:.4f}')

    torch.cuda.empty_cache()
