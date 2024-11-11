from common_imports import *
from models.vqa_model import MBart_BEiT_Model
from models.encoders import Bart_Encode_Feature, Vision_Encode_Pixel
from metrics.ScoreCalculator import ScoreCalculator
from models.Bart_Encode_Feature import Bart_tokenizer
from torch.utils.data import Dataset, DataLoader
import wandb
import gc
import torch.cuda as cuda
from torch import amp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

class VQA_Task:
    def __init__(self, config):
        # Cấu hình CUDA và distributed training
        self.setup_distributed()
        self.setup_cuda_optimization()
        
        self.save_path = os.path.join(config.TRAINING.SAVE_PATH, config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = config.TRAINING.EPOCHS
        self.patience = config.TRAINING.PATIENCE
        self.best_metric = config.TRAINING.METRIC_BEST
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.weight_decay = config.TRAINING.WEIGHT_DECAY
        self.batch_size = config.TRAINING.BATCH_SIZE
        self.num_workers = config.TRAINING.WORKERS
        self.gradient_accumulation_steps = 4
        
        # Khởi tạo autocast và scaler
        self.autocast = amp.autocast(device_type='cuda', dtype=torch.float16)
        self.scaler = amp.GradScaler()

        # Prefetch và cache cho data loading
        self.prefetch_factor = 2
        self.persistent_workers = True

        # Khởi tạo WandB chỉ trên main process
        if self.is_main_process:
            self.init_wandb(config)

        self.tokenizer = Bart_tokenizer(config.MODEL)
        self.base_model = MBart_BEiT_Model(config.MODEL).to(self.device)
        
        # Wrap model với DDP nếu sử dụng distributed training
        if self.distributed:
            self.base_model = DDP(
                self.base_model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Tăng performance
            )
        
        # Optimizer với gradient clipping
        self.optimizer = self.get_optimizer(config)
        self.scheduler = self.get_scheduler()
        self.compute_score = ScoreCalculator()

    def setup_distributed(self):
        """Thiết lập distributed training"""
        self.distributed = False
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True

        if torch.cuda.is_available():
            if 'WORLD_SIZE' in os.environ:
                self.distributed = True
                dist.init_process_group(backend='nccl')
                self.local_rank = int(os.environ['LOCAL_RANK'])
                self.world_size = dist.get_world_size()
                self.is_main_process = self.local_rank == 0
                torch.cuda.set_device(self.local_rank)

    def setup_cuda_optimization(self):
        """Tối ưu CUDA settings"""
        if torch.cuda.is_available():
            # Enable cuDNN autotuner
            cudnn.benchmark = True
            cudnn.deterministic = False
            
            # Set optimal algorithmic choices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def get_optimizer(self, config):
        """Tối ưu optimizer với Gradient Centralization và custom params"""
        params = [
            {'params': [p for n, p in self.base_model.named_parameters() if 'bias' not in n and p.requires_grad], 
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.base_model.named_parameters() if 'bias' in n and p.requires_grad], 
             'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True  # Sử dụng CUDA fused implementation
        )

    def get_data_loader(self, dataset, is_train=True):
        """Tạo optimized data loader"""
        sampler = DistributedSampler(dataset) if self.distributed else None
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None) and is_train,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=is_train
        )

    def training(self, train_dataset, valid_dataset):
        # Tạo optimized data loaders
        train_loader = self.get_data_loader(train_dataset, is_train=True)
        valid_loader = self.get_data_loader(valid_dataset, is_train=False)

        # Load checkpoint if exists
        initial_epoch, best_score = self.load_checkpoint()
        
        # Training loop với các optimization
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase với các optimization
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            valid_metrics = self.validate_epoch(valid_loader)
            
            # Log và save chỉ trên main process
            if self.is_main_process:
                self.log_metrics(epoch, train_metrics, valid_metrics)
                threshold = self.save_checkpoints(epoch, valid_metrics, best_score)
                
                # Early stopping check
                if threshold >= self.patience:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break
            
            # Synchronize processes
            if self.distributed:
                dist.barrier()
            
            # Clear memory
            self.clear_memory()

    def train_epoch(self, train_loader, epoch):
        self.base_model.train()
        
        # Sử dụng torch.cuda.Stream để overlap computation và data transfer
        train_stream = torch.cuda.Stream()
        
        metrics = {'loss': 0., 'accuracy': 0.}
        batch_count = 0
        optimizer_step = 0
        
        with tqdm(desc='Training', unit='it', total=len(train_loader), disable=not self.is_main_process) as pbar:
            for it, item in enumerate(train_loader):
                # Prefetch next batch
                if it + 1 < len(train_loader):
                    with torch.cuda.stream(train_stream):
                        next_item = next(iter(train_loader))
                
                # Forward pass với optimization
                with self.autocast:
                    logits, loss = self.base_model(
                        item['question'], 
                        item['image'],
                        item['answer']
                    )
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass với optimization
                self.scaler.scale(loss).backward()
                
                if (it + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.parameters(), 
                        max_norm=1.0
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    self.scheduler.step()
                    optimizer_step += 1
                
                metrics['loss'] += loss.item() * self.gradient_accumulation_steps
                
                # Calculate metrics periodically
                if it % 100 == 0:
                    with torch.no_grad():
                        pred_answers = self.base_model(item['question'], item['image'])
                        batch_accuracy = self.calculate_accuracy(pred_answers, item['answer'])
                        metrics['accuracy'] += batch_accuracy
                        batch_count += 1
                
                if self.is_main_process:
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']/(it+1):.4f}",
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    pbar.update()
                
                # Synchronize streams periodically
                if it % 50 == 49:
                    torch.cuda.synchronize()
                    self.clear_memory()
        
        # Calculate final metrics
        if self.distributed:
            dist.all_reduce(torch.tensor(metrics['loss']).to(self.device))
            dist.all_reduce(torch.tensor(metrics['accuracy']).to(self.device))
            metrics['loss'] /= self.world_size
            metrics['accuracy'] /= self.world_size
        
        metrics['loss'] /= len(train_loader)
        metrics['accuracy'] = metrics['accuracy'] / batch_count if batch_count > 0 else 0
        
        return metrics
