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
import traceback

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
        self.num_workers = os.cpu_count()
        self.gradient_accumulation_steps = 4
        
        # Khởi tạo autocast và scaler
        self.autocast = amp.autocast(device_type='cuda', dtype=torch.float16)
        self.scaler = amp.GradScaler()

        # Prefetch và cache cho data loading
        self.prefetch_factor = 2
        self.persistent_workers = True

        

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
        self.compute_score = ScoreCalculator()
        
        # Khởi tạo WandB chỉ trên main process
        if self.is_main_process:
            self.init_wandb(config)
        
    def init_wandb(self, config):
        """
        Khởi tạo Weights & Biases logging
        Args:
            config: Configuration object containing model and training parameters
        """
        wandb.init(
            project="VQA-Project",  # Tên project của bạn trên WandB
            name=config.MODEL.NAME,  # Tên run, sử dụng tên model
            config={
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.num_epochs,
                "weight_decay": self.weight_decay,
                "model_name": config.MODEL.NAME,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "architecture": config.MODEL.ARCHITECTURE if hasattr(config.MODEL, 'ARCHITECTURE') else "Not specified",
                "optimizer": "AdamW",
                "scheduler": "linear_warmup_decay",
                "distributed_training": self.distributed,
                "world_size": self.world_size
            }
        )
        
        # Log model graph
        if hasattr(self.base_model, 'module'):
            wandb.watch(
                self.base_model.module,
                log="gradients",
                log_freq=100,
                log_graph=True
            )
        else:
            wandb.watch(
                self.base_model,
                log="gradients",
                log_freq=100,
                log_graph=True
            )

    def log_metrics(self, epoch, train_metrics, valid_metrics):
        """
        Log metrics to WandB
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary containing training metrics
            valid_metrics: Dictionary containing validation metrics
        """
        if not self.is_main_process:
            return
            
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['accuracy'],
            "valid/loss": valid_metrics['loss'],
            "valid/accuracy": valid_metrics['accuracy'],
            "learning_rate": self.scheduler.get_last_lr()[0]
        })

    def clear_memory(self):
        """
        Clear unused memory caches
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def get_scheduler(self, train_loader):
        """
        Tạo learning rate scheduler với các biện pháp bảo vệ
        Args:
            train_loader: DataLoader cho training dataset
        Returns:
            OneCycleLR scheduler
        """
        # Tính toán tổng số steps cho mỗi epoch
        num_training_steps = len(train_loader)
        num_optimization_steps = num_training_steps // self.gradient_accumulation_steps
        
        print(f"Number of training steps per epoch: {num_training_steps}")
        print(f"Number of optimization steps per epoch: {num_optimization_steps}")
        print(f"Total steps for all epochs: {num_optimization_steps * self.num_epochs}")
        
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=num_optimization_steps * self.num_epochs,  # Sử dụng total_steps thay vì epochs
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )

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
            """
            Tạo optimized data loader với error handling
            Args:
                dataset: Dataset to load
                is_train: Whether this is for training
            Returns:
                DataLoader: Configured data loader
            """
            def collate_fn(batch):
                """
                Xử lý batch data an toàn
                """
                # Lọc bỏ các mẫu None
                batch = [item for item in batch if item is not None]
                if not batch:
                    return {}
                
                # Khởi tạo các list để chứa dữ liệu
                questions = []
                images = []
                answers = []
                
                # Thu thập dữ liệu từ batch
                for item in batch:
                    try:
                        questions.append(item.get('question', ''))
                        images.append(item.get('image', None))
                        answers.append(item.get('answer', ''))
                    except Exception as e:
                        print(f"Error processing batch item: {str(e)}")
                        print(f"Question: {item.get('question', None)}, Image: {item.get('image', None)}, Answer: {item.get('answer', None)}")
                        continue
                
                # Kiểm tra xem có đủ dữ liệu không
                if not questions or not images or not answers:
                    return {}
                    
                return {
                    'question': questions,
                    'image': torch.stack(images) if isinstance(images[0], torch.Tensor) else images,
                    'answer': answers
                }
    
            sampler = DistributedSampler(dataset) if self.distributed else None
            
            # Base loader config
            loader_config = {
                'dataset': dataset,
                'batch_size': self.batch_size,
                'shuffle': (sampler is None) and is_train,
                'sampler': sampler,
                'pin_memory': True,
                'drop_last': is_train,
                'collate_fn': collate_fn
            }
            
            # Add multiprocessing configs only if num_workers > 0
            if self.num_workers > 0:
                loader_config.update({
                    'num_workers': self.num_workers,
                    'prefetch_factor': self.prefetch_factor,
                    'persistent_workers': self.persistent_workers
                })
            
            return DataLoader(**loader_config)
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1, checkpoint['score']
        return 0, 0.0

    def save_checkpoint(self, epoch, score, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score
        }
        filename = 'best_model.pth' if is_best else 'last_model.pth'
        torch.save(checkpoint, os.path.join(self.save_path, filename))
    def calculate_accuracy(self, pred_answers, true_answers):
            """Tính toán accuracy dựa trên exact match"""
            correct = sum(1 for pred, true in zip(pred_answers, true_answers) if pred.strip().lower() == true.strip().lower())
            return correct / len(true_answers)

    def training(self, train_dataset, valid_dataset):
        train_loader = self.get_data_loader(train_dataset, is_train=True)
        valid_loader = self.get_data_loader(valid_dataset, is_train=False)
        
        initial_epoch, best_score = self.load_checkpoint(
            os.path.join(self.save_path, 'last_model.pth')
        )
        
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            # Tạo mới scheduler cho mỗi epoch
            self.scheduler = self.get_scheduler(train_loader)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            valid_metrics = self.validate_epoch(valid_loader)
            
            if self.is_main_process:
                self.log_metrics(epoch, train_metrics, valid_metrics)
                threshold = self.save_checkpoints(epoch, valid_metrics, best_score)
                
                if threshold >= self.patience:
                    print(f"Early stopping after epoch {epoch + 1}")
                    break
            
            if self.distributed:
                dist.barrier()
            
            self.clear_memory()

    def train_epoch(self, train_loader, epoch):
        self.base_model.train()
        train_stream = torch.cuda.Stream()
        
        metrics = {'loss': 0., 'accuracy': 0.}
        batch_count = 0
        step_count = 0
        
        # Tính tổng số optimization steps cho epoch hiện tại
        total_steps = len(train_loader) // self.gradient_accumulation_steps
        accumulation_counter = 0
        
        with tqdm(desc='Training', unit='it', total=len(train_loader), disable=not self.is_main_process) as pbar:
            for it, item in enumerate(train_loader):
                if not item or len(item) == 0:
                    continue
                    
                try:
                    # Đảm bảo dữ liệu ở đúng device
                    if isinstance(item['image'], torch.Tensor):
                        item['image'] = item['image'].to(self.device)
                    
                    # Prefetch next batch
                    if it + 1 < len(train_loader):
                        with torch.cuda.stream(train_stream):
                            next_item = next(iter(train_loader))
                    
                    # Forward pass với optimization và error handling
                    with self.autocast:
                        logits, loss = self.base_model(
                            item['question'], 
                            item['image'],
                            item['answer']
                        )
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass với optimization
                    self.scaler.scale(loss).backward()
                    
                    accumulation_counter += 1
                    
                    # Update model weights và scheduler mỗi gradient_accumulation_steps
                    if accumulation_counter >= self.gradient_accumulation_steps:
                        # Kiểm tra xem đã đạt đến total_steps chưa
                        if step_count < total_steps:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.base_model.parameters(), 
                                max_norm=1.0
                            )
                            
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            
                            # Step scheduler
                            self.scheduler.step()
                            step_count += 1
                        
                        accumulation_counter = 0
                    
                    metrics['loss'] += loss.item() * self.gradient_accumulation_steps
                    
                    # Calculate metrics periodically
                    if it % 100 == 0:
                        with torch.no_grad():
                            pred_answers = self.base_model(item['question'], item['image'])
                            batch_accuracy = self.calculate_accuracy(pred_answers, item['answer'])
                            metrics['accuracy'] += batch_accuracy
                            batch_count += 1
                    
                    current_lr = self.scheduler.get_last_lr()[0] if step_count < total_steps else self.scheduler.get_last_lr()[-1]
                    
                    if self.is_main_process:
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']/(it+1):.4f}",
                            'lr': f"{current_lr:.1e}",
                            'steps': f"{step_count}/{total_steps}"
                        })
                        pbar.update()
                
                except Exception as e:
                    print(f"Error in batch {it}: {str(e)}")
                    print(f"Current step: {step_count}, Total steps: {total_steps}")
                    traceback.print_exc()
                    continue
                
                # Synchronize streams periodically
                if it % 50 == 49:
                    torch.cuda.synchronize()
                    self.clear_memory()
        
        return metrics

    def get_scheduler(self, train_loader):
        """
        Tạo learning rate scheduler với các biện pháp bảo vệ
        Args:
            train_loader: DataLoader cho training dataset
        Returns:
            OneCycleLR scheduler
        """
        # Tính toán số steps chính xác
        total_steps = len(train_loader) // self.gradient_accumulation_steps
        
        print(f"\nScheduler Configuration:")
        print(f"Total batches per epoch: {len(train_loader)}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Total optimization steps: {total_steps}")
        
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0,
            anneal_strategy='cos'
        )
    def validate_epoch(self, valid_loader):
        """
        Thực hiện validation epoch
        Args:
            valid_loader: DataLoader cho validation dataset
        Returns:
            dict: Dictionary chứa các metrics validation
        """
        self.base_model.eval()
        metrics = {'loss': 0., 'accuracy': 0.}
        batch_count = 0
        
        with torch.no_grad(), \
             tqdm(desc='Validation', unit='it', total=len(valid_loader), disable=not self.is_main_process) as pbar:
            
            for it, item in enumerate(valid_loader):
                # Kiểm tra batch có dữ liệu không
                if not item or len(item) == 0:
                    continue
                    
                try:
                    # Đảm bảo dữ liệu ở đúng device
                    if isinstance(item['image'], torch.Tensor):
                        item['image'] = item['image'].to(self.device)
                    
                    # Forward pass với autocast để tối ưu memory và speed
                    with self.autocast:
                        logits, loss = self.base_model(
                            item['question'],
                            item['image'],
                            item['answer']
                        )
                    
                    metrics['loss'] += loss.item()
                    
                    # Tính accuracy cho batch
                    pred_answers = self.base_model(item['question'], item['image'])
                    batch_accuracy = self.calculate_accuracy(pred_answers, item['answer'])
                    metrics['accuracy'] += batch_accuracy
                    batch_count += 1
                    
                    if self.is_main_process:
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']/(it+1):.4f}",
                            'accuracy': f"{metrics['accuracy']/batch_count:.4f}"
                        })
                        pbar.update()
                    
                except Exception as e:
                    print(f"Error processing validation batch {it}: {str(e)}")
                    print(f"Batch data details: {item}")
                    print("Traceback details:")
                    traceback.print_exc()
                    continue
                
                # Định kỳ clear memory
                if it % 50 == 49:
                    torch.cuda.synchronize()
                    self.clear_memory()
        
        # Tính final metrics
        if batch_count > 0:  # Chỉ tính metrics nếu có ít nhất một batch thành công
            if self.distributed:
                # Gather metrics từ tất cả processes
                dist.all_reduce(torch.tensor(metrics['loss']).to(self.device))
                dist.all_reduce(torch.tensor(metrics['accuracy']).to(self.device))
                metrics['loss'] /= self.world_size
                metrics['accuracy'] /= self.world_size
            
            metrics['loss'] /= len(valid_loader)
            metrics['accuracy'] /= batch_count
        
        return metrics
    def save_checkpoint_to_wandb(self, checkpoint, name):
        """
        Save checkpoint to WandB
        Args:
            checkpoint: Dictionary containing model checkpoint
            name: Name of the checkpoint file
        """
        if not self.is_main_process:
            return
    
        # Create artifact
        artifact = wandb.Artifact(
            name=f"vqa_checkpoint_{name}",
            type="model",
            metadata={
                "epoch": checkpoint['epoch'],
                "score": checkpoint['score']
            }
        )
    
        # Save checkpoint temporarily
        temp_path = os.path.join(self.save_path, f"temp_{name}_checkpoint.pth")
        torch.save(checkpoint, temp_path)
    
        # Add checkpoint to artifact
        artifact.add_file(temp_path)
        wandb.log_artifact(artifact)
    
        # Remove temporary file
        os.remove(temp_path)
    
    def load_checkpoint_from_wandb(self, artifact_name=None):
        """
        Load checkpoint from WandB
        Args:
            artifact_name: Optional specific artifact name to load
        Returns:
            Tuple of (starting_epoch, best_score) or (0, 0.0) if no checkpoint found
        """
        if not self.is_main_process:
            return 0, 0.0
    
        try:
            # If no specific artifact name, try to get the latest
            if artifact_name is None:
                artifact = wandb.use_artifact('vqa_checkpoint_best:latest')
            else:
                artifact = wandb.use_artifact(artifact_name)
    
            # Download artifact
            artifact_dir = artifact.download(self.save_path)
            checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
    
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
            return checkpoint['epoch'] + 1, checkpoint['score']
    
        except Exception as e:
            print(f"No WandB checkpoint found: {e}")
            return 0, 0.0    
    def save_checkpoints(self, epoch, valid_metrics, best_score):
        current_score = valid_metrics['accuracy'] if self.best_metric == 'accuracy' else -valid_metrics['loss']
        threshold = 0
        
        # Lưu model state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': current_score
        }
        
        self.save_checkpoint(epoch, current_score)
        
        # Kiểm tra xem có phải best model không
        if current_score > best_score:
            # Lưu best checkpoint
            self.save_checkpoint(epoch, current_score, is_best=True)
            
            # Save best checkpoint to WandB
            if self.is_main_process:
                self.save_checkpoint_to_wandb(checkpoint, 'best')
            
            best_score = current_score
            threshold = 0
        else:
            threshold += 1
        
        return threshold
