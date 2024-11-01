# trainer/vqa_task.py
from common_imports import *
from models.vqa_model import MBart_BEiT_Model
from models.encoders import Bart_Encode_Feature, Vision_Encode_Pixel
from metrics.ScoreCalculator import ScoreCalculator

class VQA_Task:
    def __init__(self, config):
        self.save_path = os.path.join(config.SAVE_PATH,config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = config.TRAINING.EPOCHS
        self.patience = config.TRAINING.PATIENCE
        self.best_metric = config.TRAINING.METRIC_BEST
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.weight_decay = config.TRAINING.WEIGHT_DECAY
        self.batch_size = config.TRAINING.BATCH_SIZE
        self.num_workers = config.TRAINING.WORKERS

        self.base_model = MBart_BEiT_Model(config.MODEL).to(self.device)
        self.optimizer = optim.Adam(
            self.base_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 0.9 ** epoch
        )
        self.compute_score = ScoreCalculator()

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

    def training_step(self, batch):
        logits, loss = self.base_model(
            batch['question'],
            batch['image'],
            batch['answer']
        )
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            loss = self.base_model(
                batch['question'],
                batch['image'],
                batch['answer']
            )[1]
            pred_answers = self.base_model(batch['question'], batch['image'])
            return loss, pred_answers, batch['answer']

    def training(self, train_loader, valid_loader):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        initial_epoch, best_score = self.load_checkpoint(
            os.path.join(self.save_path, 'last_model.pth')
        )

        threshold = 0
        self.base_model.train()

        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            train_loss = 0.0
            valid_loss = 0.0
            
            # Training loop
            with tqdm(desc=f'Epoch {epoch + 1} - Training', total=len(train_loader)) as pbar:
                for it, batch in enumerate(train_loader):
                    loss = self.training_step(batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    train_loss += loss.item()
                    pbar.set_postfix({
                        'loss': f"{train_loss/(it+1):.4f}",
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    pbar.update()
                    
            self.scheduler.step()
            train_loss /= len(train_loader)

            # Validation loop  
            metrics = {'wups': 0.0, 'em': 0.0, 'f1': 0.0, 'cider': 0.0}
            
            with tqdm(desc=f'Epoch {epoch + 1} - Validation', total=len(valid_loader)) as pbar:
                for batch in valid_loader:
                    loss, preds, targets = self.validation_step(batch)
                    valid_loss += loss.item()
                    
                    metrics['wups'] += self.compute_score.wup(targets, preds)
                    metrics['em'] += self.compute_score.em(targets, preds) 
                    metrics['f1'] += self.compute_score.f1_token(targets, preds)
                    metrics['cider'] += self.compute_score.cider_score(targets, preds)
                    
                    pbar.update()

            valid_loss /= len(valid_loader)
            metrics = {k: v/len(valid_loader) for k,v in metrics.items()}
            
            self.log_metrics(epoch, train_loss, valid_loss, metrics)
            
            score = metrics[self.best_metric]
            self.save_checkpoint(epoch, score, is_best=score > best_score)

            if score > best_score:
                best_score = score
                print(f"Saved best model with {self.best_metric} of {score:.4f}")
                threshold = 0
            else:
                threshold += 1

            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    def log_metrics(self, epoch, train_loss, valid_loss, metrics):
        log_str = (f"Epoch {epoch + 1} - "
                  f"Training loss: {train_loss:.4f} - "
                  f"Valid loss: {valid_loss:.4f} - "
                  f"Valid metrics: {metrics}")
        
        print(log_str)
        with open(os.path.join(self.save_path, 'log.txt'), 'a') as f:
            f.write(log_str + '\n')

    def get_predict(self, test_loader):
        # Load best model
        self.load_checkpoint(os.path.join(self.save_path, 'best_model.pth'))
        
        print("Predicting...")
        results = {
            'img_path': [],
            'question': [],
            'ground_truth': [],
            'predicts': []
        }

        self.base_model.eval()
        metrics = {'wups': 0.0, 'em': 0.0, 'f1': 0.0, 'cider': 0.0}

        with torch.no_grad():
            for batch in tqdm(test_loader):
                pred_answers = self.base_model(batch['question'], batch['image'])
                
                results['predicts'].extend(pred_answers)
                results['img_path'].extend(batch['image'])
                results['question'].extend(batch['question'])
                results['ground_truth'].extend(batch['answer'])

        # Calculate metrics
        gts = results['ground_truth']
        preds = results['predicts']
        
        metrics['wups'] = self.compute_score.wup(gts, preds)
        metrics['em'] = self.compute_score.em(gts, preds)
        metrics['f1'] = self.compute_score.f1_token(gts, preds)
        metrics['cider'] = self.compute_score.cider_score(gts, preds)

        print("\nTest metrics:", metrics)

        # Save results
        df = pd.DataFrame(results)
        save_path = os.path.join(self.save_path, 'result.csv')
        df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
