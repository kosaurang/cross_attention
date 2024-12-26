import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np

# TPU-specific imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from tqdm import tqdm
import wandb

# Giả định các import từ project của bạn
from common_imports import *
from models.vqa_model import MBart_BEiT_Model
from models.encoders import Bart_Encode_Feature, Vision_Encode_Pixel
from metrics.ScoreCalculator import ScoreCalculator
from models.Bart_Encode_Feature import Bart_tokenizer    
from types import SimpleNamespace

class VQAConfig:
    def __init__(self, yaml_config):
        # Training config
        self.TRAINING = SimpleNamespace(
            CHECKPOINT_PATH=yaml_config['TRAINING']['CHECKPOINT_PATH'],
            LEARNING_RATE=yaml_config['TRAINING']['LEARNING_RATE'], 
            WEIGHT_DECAY=yaml_config['TRAINING']['WEIGHT_DECAY'],
            EPOCHS=yaml_config['TRAINING']['EPOCHS'],
            BATCH_SIZE=yaml_config['TRAINING']['BATCH_SIZE'],
            METRIC_BEST=yaml_config['TRAINING']['METRIC_BEST'],
            PATIENCE=yaml_config['TRAINING']['PATIENCE'],
            SAVE_PATH=yaml_config['TRAINING']['SAVE_PATH']
        )
        
        # Model config với cấu trúc phù hợp cho MBart_BEiT_Model
        self.MODEL = SimpleNamespace(
            NAME=yaml_config['MODEL']['NAME'],
            DEVICE=yaml_config['MODEL']['DEVICE'],
            VISION_EMBEDDING=SimpleNamespace(
                PRETRAINED_NAME=yaml_config['MODEL']['VISION_EMBEDDING']['PRETRAINED_NAME'],
                D_PRETRAINED_FEATURE=yaml_config['MODEL']['VISION_EMBEDDING']['D_PRETRAINED_FEATURE']
            ),
            TEXT_EMBEDDING=SimpleNamespace(
                PRETRAINED_NAME=yaml_config['MODEL']['TEXT_EMBEDDING']['PRETRAINED_NAME']
            ),
            TOKENIZER=SimpleNamespace(
                PADDING=yaml_config['MODEL']['TOKENIZER']['PADDING'],
                MAX_INPUT_LENGTH=yaml_config['MODEL']['TOKENIZER']['MAX_INPUT_LENGTH'],
                MAX_TARGET_LENGTH=yaml_config['MODEL']['TOKENIZER']['MAX_TARGET_LENGTH'],
                TRUNCATION=yaml_config['MODEL']['TOKENIZER']['TRUNCATION'],
                RETURN_ATTENTION_MASK=yaml_config['MODEL']['TOKENIZER']['RETURN_ATTENTION_MASK']
            ),
            GENERATOR=SimpleNamespace(
                MAX_LENGTH=yaml_config['MODEL']['GENERATOR']['MAX_LENGTH'],
                MIN_LENGTH=yaml_config['MODEL']['GENERATOR']['MIN_LENGTH'],
                NUM_BEAMS=yaml_config['MODEL']['GENERATOR']['NUM_BEAMS'],
                LENGTH_PENALTY=yaml_config['MODEL']['GENERATOR']['LENGTH_PENALTY'],
                NO_REPEAT_NGRAM_SIZE=yaml_config['MODEL']['GENERATOR']['NO_REPEAT_NGRAM_SIZE'],
                EARLY_STOPPING=yaml_config['MODEL']['GENERATOR']['EARLY_STOPPING']
            )
        )
        
        
class VQA_Task_TPU:
    def __init__(self, config):
        # TPU-specific setup
        self.device = xm.xla_device()
        self.world_size = xm.xrt_world_size()
        self.global_rank = xm.get_ordinal()
        self.is_main_process = (self.global_rank == 0)
    
        # Lưu trữ config
        self.config = config
        model_config = SimpleNamespace()
        model_config.MODEL = SimpleNamespace(
        NAME=self.config.MODEL.NAME,
        DEVICE=self.config.MODEL.DEVICE,
        VISION_EMBEDDING=self.config.MODEL.VISION_EMBEDDING,
        TEXT_EMBEDDING=self.config.MODEL.TEXT_EMBEDDING,
        TOKENIZER=self.config.MODEL.TOKENIZER,
        GENERATOR=self.config.MODEL.GENERATOR
        )
        
        # Thư mục lưu checkpoint
        os.makedirs(config.TRAINING.SAVE_PATH, exist_ok=True)
        self.save_path = os.path.join(config.TRAINING.SAVE_PATH, config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        os.makedirs(self.save_path, exist_ok=True)
    
        # Các siêu tham số huấn luyện
        self.num_epochs = config.TRAINING.EPOCHS
        self.patience = config.TRAINING.PATIENCE
        self.best_metric = config.TRAINING.METRIC_BEST
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.batch_size = config.TRAINING.BATCH_SIZE
        self.gradient_accumulation_steps = 4
    
        # Tokenizer và model
        self.tokenizer = Bart_tokenizer(self.config.MODEL)
        self.base_model = MBart_BEiT_Model(model_config)
        self.base_model = self.base_model.to(self.device)
        #self.base_model = MBart_BEiT_Model(config.MODEL).to(self.device)
        #self.base_model = xm.send_cpu_data_to_device(self.base_model, self.device)
        
        # Optimizer
        self.optimizer = self.get_optimizer()
        
        # Metrics
        self.compute_score = ScoreCalculator()
        
        # Khởi tạo Weights & Biases
        if self.is_main_process:
            wandb.init(
                project="VQA_TPU_Project",
                name=config.MODEL.NAME,
                config=vars(config)
            )
    
    def get_optimizer(self):
        """Cấu hình optimizer cho TPU"""
        return torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.TRAINING.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
    
    def get_tpu_loader(self, dataset, is_train=True):
        try:
            def collate_fn(batch):
                batch = [item for item in batch if item is not None]
                if not batch:
                    return None  # Trả về None thay vì từ điển rỗng
                
                questions = [item['question'] for item in batch]
                images = [item['image'] for item in batch]
                answers = [item['answer'] for item in batch]
                
                return {
                    'question': questions,
                    'image': torch.stack(images),
                    'answer': answers
                }
            
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=is_train,
                collate_fn=collate_fn,
                drop_last=is_train
            )
            
            # In ra thông tin về loader để kiểm tra
            print(f"Loader created with {len(loader)} batches")
            return loader
        except Exception as e:
            print(f"Loader creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_accuracy(self, predictions, targets):
            """Tính độ chính xác"""
            correct = sum(
                pred.strip().lower() == target.strip().lower() 
                for pred, target in zip(predictions, targets)
            )
            return correct / len(targets)
    
    def train_epoch(self, train_loader, epoch):
                """Huấn luyện một epoch"""
                self.base_model.train()
                metrics = {'loss': 0., 'accuracy': 0.}
                
                para_loader = pl.ParallelLoader(train_loader, [self.device]).per_device_loader(self.device)
                
                for it, batch in enumerate(para_loader):
                    if not batch:
                        continue
                    
                    try:
                        # Forward và backward
                        logits, loss = self.base_model(
                            batch['question'], 
                            batch['image'],
                            batch['answer']
                        )
                        
                        loss = loss / self.gradient_accumulation_steps
                        loss.backward()
                        
                        # Cập nhật trọng số
                        if (it + 1) % self.gradient_accumulation_steps == 0:
                            xm.optimizer_step(self.optimizer)
                            xm.mark_step()
                            self.optimizer.zero_grad()
                        
                        # Tính metrics
                        metrics['loss'] += loss.item()
                        
                        if it % 100 == 0:
                            pred_answers = self.base_model(batch['question'], batch['image'])
                            batch_accuracy = self.calculate_accuracy(pred_answers, batch['answer'])
                            metrics['accuracy'] += batch_accuracy
                    
                    except Exception as e:
                        xm.master_print(f"Lỗi tại batch {it}: {e}")
                        continue
                
                return metrics
        
    def validate_epoch(self, valid_loader):
                """Đánh giá trên tập validation"""
                self.base_model.eval()
                metrics = {'loss': 0., 'accuracy': 0.}
                
                para_loader = pl.ParallelLoader(valid_loader, [self.device]).per_device_loader(self.device)
                
                with torch.no_grad():
                    for it, batch in enumerate(para_loader):
                        if not batch:
                            continue
                        
                        try:
                            logits, loss = self.base_model(
                                batch['question'], 
                                batch['image'],
                                batch['answer']
                            )
                            
                            metrics['loss'] += loss.item()
                            
                            pred_answers = self.base_model(batch['question'], batch['image'])
                            batch_accuracy = self.calculate_accuracy(pred_answers, batch['answer'])
                            metrics['accuracy'] += batch_accuracy
                        
                        except Exception as e:
                            xm.master_print(f"Lỗi validation ở batch {it}: {e}")
                            continue
                
                return metrics
    
    def save_checkpoint(self, epoch, metrics):
                """Lưu checkpoint"""
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': metrics
                }
                
                checkpoint_path = os.path.join(self.save_path, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                
                # Log checkpoint lên wandb
                if self.is_main_process:
                    wandb.save(checkpoint_path)
        
    def training(self, train_dataset, valid_dataset):
                """Vòng lặp huấn luyện chính"""
                train_loader = self.get_tpu_loader(train_dataset, is_train=True)
                valid_loader = self.get_tpu_loader(valid_dataset, is_train=False)
                
                best_metric = 0
                patience_counter = 0
                
                for epoch in range(self.num_epochs):
                    # Huấn luyện
                    train_metrics = self.train_epoch(train_loader, epoch)
                    valid_metrics = self.validate_epoch(valid_loader)
                    
                    # Log metrics
                    if self.is_main_process:
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_metrics['loss'],
                            'train_accuracy': train_metrics['accuracy'],
                            'valid_loss': valid_metrics['loss'],
                            'valid_accuracy': valid_metrics['accuracy']
                        })
                    
                    # Lưu checkpoint
                    self.save_checkpoint(epoch, valid_metrics)
                    
                    # Kiểm tra điều kiện early stopping
                    current_metric = valid_metrics['accuracy']
                    if current_metric > best_metric:
                        best_metric = current_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        xm.master_print(f"Early stopping tại epoch {epoch}")
                        break
