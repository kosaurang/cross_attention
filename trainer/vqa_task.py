from common_imports import *
from models.vqa_model import MBart_BEiT_Model
from models.encoders import Bart_Encode_Feature, Vision_Encode_Pixel
from metrics.ScoreCalculator import ScoreCalculator
from models.Bart_Encode_Feature import Bart_tokenizer
from torch.utils.data import Dataset, DataLoader
import wandb
import gc
import torch.cuda as cuda

class VQA_Task:
    def __init__(self, config):
        self.save_path = os.path.join(config.TRAINING.SAVE_PATH, config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = config.TRAINING.EPOCHS
        self.patience = config.TRAINING.PATIENCE
        self.best_metric = config.TRAINING.METRIC_BEST
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.weight_decay = config.TRAINING.WEIGHT_DECAY
        self.batch_size = config.TRAINING.BATCH_SIZE
        self.num_workers = config.TRAINING.WORKERS
        self.gradient_accumulation_steps = 4  # Thêm gradient accumulation để giảm memory usage

        # Khởi tạo WandB
        wandb.init(
            project="vqa-training",
            config={
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "model": config.MODEL.NAME,
                "epochs": self.num_epochs
            }
        )

        self.tokenizer = Bart_tokenizer(config.MODEL)
        self.base_model = MBart_BEiT_Model(config.MODEL).to(self.device)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.9 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.compute_score = ScoreCalculator()

    def clear_memory(self):
        """Hàm giải phóng memory"""
        gc.collect()
        cuda.empty_cache()
        
    def calculate_accuracy(self, pred_answers, true_answers):
        """Tính toán accuracy dựa trên exact match"""
        correct = sum(1 for pred, true in zip(pred_answers, true_answers) if pred.strip().lower() == true.strip().lower())
        return correct / len(true_answers)

    def training(self, train_dataset, valid_dataset):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Load checkpoint if exists
        initial_epoch = 0
        best_score = 0.
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initial_epoch = checkpoint['epoch'] + 1
            print(f"Continue training from epoch {initial_epoch}")
        
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']

        threshold = 0
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            print(f'\nEpoch {epoch + 1}/{self.num_epochs + initial_epoch}')
            
            # Training phase
            self.base_model.train()
            train_loss = 0.
            train_accuracy = 0.
            batch_count = 0
            
            with tqdm(desc='Training', unit='it', total=len(train)) as pbar:
                for it, item in enumerate(train):
                    # Gradient accumulation steps
                    with torch.amp.autocast():
                        logits, loss = self.base_model(item['question'], item['image'], item['answer'])
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (it + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    
                    train_loss += loss.item() * self.gradient_accumulation_steps
                    
                    # Calculate training accuracy periodically
                    if it % 100 == 0:
                        with torch.no_grad():
                            pred_answers = self.base_model(item['question'], item['image'])
                            batch_accuracy = self.calculate_accuracy(pred_answers, item['answer'])
                            train_accuracy += batch_accuracy
                            batch_count += 1
                    
                    pbar.set_postfix({
                        'loss': f"{train_loss/(it+1):.4f}",
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    pbar.update()
                    
                    # Clear memory periodically
                    if it % 50 == 0:
                        self.clear_memory()

            self.scheduler.step()
            avg_train_loss = train_loss / len(train)
            avg_train_accuracy = train_accuracy / batch_count if batch_count > 0 else 0

            # Validation phase
            self.base_model.eval()
            valid_metrics = {
                'loss': 0.,
                'em': 0.,
                'wups': 0.,
                'f1': 0.,
                'cider': 0.,
                'accuracy': 0.
            }
            
            with torch.no_grad():
                with tqdm(desc='Validation', unit='it', total=len(valid)) as pbar:
                    for it, item in enumerate(valid):
                        # Forward pass
                        _, loss = self.base_model(item['question'], item['image'], item['answer'])
                        pred_answers = self.base_model(item['question'], item['image'])
                        
                        # Calculate metrics
                        valid_metrics['loss'] += loss.item()
                        valid_metrics['wups'] += self.compute_score.wup(item['answer'], pred_answers)
                        valid_metrics['em'] += self.compute_score.em(item['answer'], pred_answers)
                        valid_metrics['f1'] += self.compute_score.f1_token(item['answer'], pred_answers)
                        valid_metrics['cider'] += self.compute_score.cider_score(item['answer'], pred_answers)
                        valid_metrics['accuracy'] += self.calculate_accuracy(pred_answers, item['answer'])
                        
                        pbar.update()
                        
                        # Clear memory periodically
                        if it % 50 == 0:
                            self.clear_memory()

            # Calculate average metrics
            for key in valid_metrics:
                valid_metrics[key] /= len(valid)

            # Log metrics to WandB
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_accuracy,
                'valid_loss': valid_metrics['loss'],
                'valid_accuracy': valid_metrics['accuracy'],
                'valid_em': valid_metrics['em'],
                'valid_wups': valid_metrics['wups'],
                'valid_f1': valid_metrics['f1'],
                'valid_cider': valid_metrics['cider'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })

            # Print metrics
            print(f"\nTraining loss: {avg_train_loss:.4f} - Training accuracy: {avg_train_accuracy:.4f}")
            print(f"Validation - Loss: {valid_metrics['loss']:.4f} - Accuracy: {valid_metrics['accuracy']:.4f} - EM: {valid_metrics['em']:.4f} - WUPS: {valid_metrics['wups']:.4f} - F1: {valid_metrics['f1']:.4f} - CIDEr: {valid_metrics['cider']:.4f}")

            # Save metrics to log file
            with open(os.path.join(self.save_path, 'log.txt'), 'a') as file:
                file.write(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f} - Training accuracy: {avg_train_accuracy:.4f} - "
                          f"Valid loss: {valid_metrics['loss']:.4f} - Valid accuracy: {valid_metrics['accuracy']:.4f} - "
                          f"Valid wups: {valid_metrics['wups']:.4f} - Valid em: {valid_metrics['em']:.4f} - "
                          f"Valid f1: {valid_metrics['f1']:.4f} - Valid cider: {valid_metrics['cider']:.4f}\n")

            # Get current score based on best_metric
            score = valid_metrics[self.best_metric]

            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score
            }
            
            # Save last model
            torch.save(checkpoint, os.path.join(self.save_path, 'last_model.pth'))

            # Save best model
            if score > best_score:
                best_score = score
                torch.save(checkpoint, os.path.join(self.save_path, 'best_model.pth'))
                print(f"Saved best model with {self.best_metric} of {score:.4f}")
                threshold = 0
            else:
                threshold += 1

            # Early stopping
            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

            # Clear memory at end of epoch
            self.clear_memory()

        wandb.finish()

    def get_predict(self, test_dataset):
        """Rest of the prediction code remains the same, but add accuracy metric"""
        print("Loading best model...")
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available")

        if not test_dataset:
            raise Exception("Not found test dataset")

        print(f"[INFO] Test size: {len(test_dataset)}")
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        print("Predicting...")
        img_path = []
        quests = []
        gts = []
        preds = []
        
        self.base_model.eval()
        with torch.no_grad():
            for it, item in enumerate(tqdm(test_loader)):
                answers = self.base_model(item['question'], item['image'])
                preds.extend(answers)
                img_path.extend(item['image'])
                quests.extend(item['question'])
                gts.extend(item['answer'])
                
                if it % 50 == 0:
                    self.clear_memory()

        # Calculate all metrics including accuracy
        test_accuracy = self.calculate_accuracy(preds, gts)
        test_wups = self.compute_score.wup(gts, preds)
        test_em = self.compute_score.em(gts, preds)
        test_f1 = self.compute_score.f1_token(gts, preds)
        test_cider = self.compute_score.cider_score(gts, preds)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"WUPS: {test_wups:.4f}")
        print(f"EM: {test_em:.4f}")
        print(f"F1: {test_f1:.4f}")
        print(f"CIDEr: {test_cider:.4f}")

        data = {
            "img_path": img_path,
            "question": quests,
            "ground_truth": gts,
            "predicts": preds
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_path, 'result.csv'), index=False)
        print(f"Save result to: {os.path.join(self.save_path,'result.csv')}")
