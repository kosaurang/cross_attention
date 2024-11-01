# models/encoders.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BeitImageProcessor, AutoConfig
from PIL import Image
import os

class Bart_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Bart_Encode_Feature, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        self.padding = config.TOKENIZER.PADDING
        self.max_input_length = config.TOKENIZER.MAX_INPUT_LENGTH
        self.max_target_length = config.TOKENIZER.MAX_TARGET_LENGTH 
        self.truncation = config.TOKENIZER.TRUNCATION
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_text, text_pair=None, answers=None):
        encoded_inputs = self.tokenizer(
            input_text, text_pair,
            padding=self.padding,
            max_length=self.max_input_length,
            truncation=self.truncation,
            return_tensors='pt'
        ).to(self.device)

        if answers is not None:
            encoded_targets = self.tokenizer(
                answers,
                padding=self.padding, 
                max_length=self.max_target_length,
                truncation=self.truncation,
                return_tensors='pt'
            ).to(self.device)
            encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'labels': encoded_targets.input_ids,
                'decoder_attention_mask': encoded_targets.attention_mask
            }
        
        return {
            'input_ids': encoded_inputs.input_ids,
            'attention_mask': encoded_inputs.attention_mask
        }

class Vision_Encode_Pixel(nn.Module):
    def __init__(self, config):
        super(Vision_Encode_Pixel, self).__init__()
        self.preprocessor = BeitImageProcessor.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images):
        processed_images = self.preprocessor(
            images=[Image.open(img_path).convert('RGB') for img_path in images],
            return_tensors="pt"
        ).to(self.device)
        return processed_images.pixel_values

# models/vqa_model.py
class MBart_BEiT_Model(nn.Module):
    def __init__(self, config):
        super(MBart_BEiT_Model, self).__init__()
        self.config = config
        self.vision_encoder_pixel = Vision_Encode_Pixel(config)
        self.text_encoder = Bart_Encode_Feature(config) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        self.embedding = Bart_Embedding(config)

        self.generator_args = {
            'max_length': config.GENERATOR.MAX_LENGTH,
            'min_length': config.GENERATOR.MIN_LENGTH,
            'num_beams': config.GENERATOR.NUM_BEAMS,
            'length_penalty': config.GENERATOR.LENGTH_PENALTY,
            'no_repeat_ngram_size': config.GENERATOR.NO_REPEAT_NGRAM_SIZE,
            'early_stopping': config.GENERATOR.EARLY_STOPPING,
        }

    def forward(self, questions, images, labels=None):
        encoding_pixel = self.vision_encoder_pixel(images)
        inputs = self.text_encoder(questions, None, labels)
        inputs.update({'pixel_values': encoding_pixel})

        if labels is not None:
            outputs = self.embedding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids = self.embedding.generate(**inputs, **self.generator_args)
            pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens

# trainer/vqa_task.py
class VQA_Task:
    def __init__(self, config):
        self.save_path = os.path.join(config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
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
