import os
import copy
import torch
import numpy as np
from tqdm import tqdm

from .arrays import batch_to_device, to_np
from .timer import Timer

class EMA:
    """Exponential Moving Average (모델 파라미터 안정화)"""
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_average(ma_params.data, current_params.data)

class Trainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,   # ✅ 추가됨
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=100,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,                 # ✅ 추가됨
        bucket=None,
        device=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        # Device 자동 설정
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Trainer 기본 설정
        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.n_reference = n_reference
        self.bucket = bucket

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        )

        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.logdir = results_folder

        self.step = 0
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self, n_train_steps):
        timer = Timer()
        data_iter = iter(self.dataloader)

        for step in tqdm(range(n_train_steps)):
            self.model.train()
            total_loss = 0.0

            for i in range(self.gradient_accumulate_every):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                batch = batch_to_device(batch, self.device)
                x, cond_tensor, cond_mask = batch     # ← unpack (중요!)
                loss, infos = self.model.loss(x, cond_tensor, cond_mask)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                total_loss += loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save(self.step)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{k}: {v:8.4f}' for k, v in infos.items()])
                print(f"{self.step}: {total_loss:8.4f} | {infos_str} | t: {timer():8.4f}")

            self.step += 1

    def save(self, epoch):
        # ✅ 폴더 자동 생성 추가
        os.makedirs(self.logdir, exist_ok=True)
        
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f"[Trainer] Saved model to {savepath}")

    def load(self, epoch):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


