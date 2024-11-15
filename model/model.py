import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed
from pathlib import Path
import soundfile as sf

from .generator import LiSenNet
from .discriminator import Discriminator


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.generator = LiSenNet(**config['model_config'])
        self.discriminator = Discriminator()
        self.current_traning_step = -1

        # Note: we use manual optimization for GAN-like training,
        # in order to optimize both generator and discriminator in one batch.
        self.automatic_optimization = False 
    
    def forward(self, batch):
        src, tgt, length, _ = batch
        results = self.generator(src, tgt)
        return results

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        g_opt, d_opt = self.optimizers()

        src, tgt, length, _ = batch
        results = self.generator(src, tgt)

        # optimize disciminator
        d_opt.zero_grad()
        d_loss, pesq_score = self.cal_d_loss(results)
        self.manual_backward(d_loss)
        self.clip_gradients(d_opt, gradient_clip_val=self.config['gradient_clip_val'], gradient_clip_algorithm='norm')
        d_opt.step()

        
        # optimize generator
        g_opt.zero_grad()
        g_loss = self.cal_g_loss(results)
        self.manual_backward(g_loss)
        self.clip_gradients(g_opt, gradient_clip_val=self.config['gradient_clip_val'], gradient_clip_algorithm='norm')
        g_opt.step()

        self.log_dict({'train_g_loss': g_loss, 'train_d_loss': d_loss}, on_step=True, on_epoch=True ,prog_bar=False)
        self.current_traning_step += 1
        # No need to return loss under manual optimization mode.

    def on_train_epoch_end(self):
        # Manually update scheduler under manual optimization mode.
        g_sch, d_sch = self.lr_schedulers()
        g_sch.step()
        d_sch.step()

    def validation_step(self, batch, batch_idx):
        src, tgt, length, _ = batch
        results = self.generator(src, tgt)

        d_loss, pesq_score = self.cal_d_loss(results)
        g_loss = self.cal_g_loss(results)

        if pesq_score is not None:
            self.log_dict({'val_g_loss': g_loss, 'val_d_loss': d_loss, 'pesq_score': pesq_score}, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_end(self,):
        # save ckpt when validation epoch finished
        if not self.trainer.sanity_checking:
            epoch = self.current_epoch
            step = self.current_traning_step
            pesq_score = self.trainer.callback_metrics['pesq_score']
            ckpt_name = f'epoch={epoch}-step={step}-pesq={pesq_score:.2f}.ckpt'
            self.trainer.save_checkpoint(self.config['ckpt_dir'] / ckpt_name)
    
    def test_step(self, batch, batch_idx):
        src, tgt, length, names = batch
        results = self.generator(src, tgt)

        est = results['est']
        assert est.size(0) == 1
        est = est.squeeze(0).cpu().numpy()
        tgt = tgt.squeeze(0).cpu().numpy()

        pesq_score = pesq(16000, tgt, est, 'wb')
        stoi_score = stoi(tgt, est, 16000)
        self.log_dict({'test_pesq': pesq_score, 'test_stoi': stoi_score}, on_step=False, on_epoch=True)

        if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
            est = est / (np.max(np.abs(est)) + 1e-5)
            sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=16000)
    
    def on_test_epoch_end(self,):
        # print(f'PESQ: {self.trainer.callback_metrics["test_pesq"]:.2f}')
        # print(f'STOI: {self.trainer.callback_metrics["test_stoi"]:.3f}')
        pass

    def on_save_checkpoint(self, ckpt):
        ckpt['current_traning_step'] = self.current_traning_step
    
    def on_load_checkpoint(self, ckpt):
        self.current_traning_step = ckpt['current_traning_step']

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.generator.parameters(), **self.config['g_opt'])
        d_opt = torch.optim.AdamW(self.discriminator.parameters(), **self.config['d_opt'])

        # Actually, the keys 'interval' and 'frequency' will be ignored under manual optimization mode.
        # We just reserve them for the automatic optimization alternative.
        g_sch = {
            'scheduler': torch.optim.lr_scheduler.StepLR(g_opt, **self.config['g_sch']), 
            'interval': 'epoch',
            'frequency': 1,
        }
        d_sch = {
            'scheduler': torch.optim.lr_scheduler.StepLR(d_opt, **self.config['d_sch']), 
            'interval': 'epoch',
            'frequency': 1,
        }

        return [g_opt, d_opt], [g_sch, d_sch]
    
    def cal_g_loss(self, results):

        g_loss = 0
        complex_loss = F.mse_loss(results['est_spec'].real, results['tgt_spec'].real) + F.mse_loss(results['est_spec'].imag, results['tgt_spec'].imag)
        g_loss += complex_loss * self.config['weights']['complex']

        mag_loss = F.mse_loss(results['est_mag'], results['tgt_mag'])
        g_loss += mag_loss * self.config['weights']['mag']

        fake = self.discriminator(results['tgt_mag'], results['est_mag'])
        adv_loss = F.mse_loss(fake[-1], fake[-1].new_ones(fake[-1].size()))
        g_loss += adv_loss * self.config['weights']['adv']

        return g_loss
    
    def cal_d_loss(self, results):
        device = results["est"].device
        est = results["est"].detach()
        tgt = results["tgt"]

        est_list = list(est.cpu().numpy())
        tgt_list = list(tgt.cpu().numpy())

        pesq_score, scaled_score = self.batch_pesq(tgt_list, est_list)
        if scaled_score is not None:
            scaled_score = scaled_score.to(device)
            fake = self.discriminator(results['tgt_mag'], results['est_mag'].detach())
            real = self.discriminator(results['tgt_mag'], results['tgt_mag'])
            d_loss = F.mse_loss(real[-1], real[-1].new_ones(real[-1].size())) + \
                        F.mse_loss(fake[-1].flatten(), scaled_score)
            pesq_score = pesq_score.mean()
        else:
            real = self.discriminator(results['tgt_mag'], results['tgt_mag'])
            d_loss = F.mse_loss(real[-1], real[-1].new_ones(real[-1].size()))
            pesq_score = None
        
        return d_loss, pesq_score

    @staticmethod
    def pesq_loss(clean, noisy, sr=16000):
        try:
            pesq_score = pesq(sr, clean, noisy, "wb")
        except:
            # error can happen due to silent period
            pesq_score = -1
        return pesq_score
    
    def batch_pesq(self, clean, noisy):
        pesq_score = Parallel(n_jobs=min(4, len(clean)))(
            delayed(self.pesq_loss)(c, n) for c, n in zip(clean, noisy)
        )
        pesq_score = np.array(pesq_score)
        if -1 in pesq_score:
            return None, None
        scaled_score = (pesq_score - 1) / 3.5
        scaled_score = np.clip(scaled_score, 0, 1)
        return torch.FloatTensor(pesq_score), torch.FloatTensor(scaled_score)
    
    
    



