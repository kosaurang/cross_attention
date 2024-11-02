class VQA_Task:
    def __init__(self, config):
        self.save_path = os.path.join(config.TRAINING.SAVE_PATH,config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
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
        try:
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.base_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if self.device.type == 'cuda':
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                
                epoch = checkpoint.get('epoch', -1)
                score = checkpoint.get('score', 0.0)
                print(f"Loaded checkpoint from epoch {epoch + 1} with score {score:.4f}")
                return epoch + 1, score
            else:
                print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
                return 0, 0.0
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return 0, 0.0

    def save_checkpoint(self, epoch, score, is_best=False):
        try:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score
            }

            last_checkpoint_path = os.path.join(self.save_path, 'last_model.pth')
            torch.save(checkpoint, last_checkpoint_path)
            print(f"Saved checkpoint for epoch {epoch + 1}")

            if is_best:
                best_checkpoint_path = os.path.join(self.save_path, 'best_model.pth')
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Saved best model checkpoint with score {score:.4f}")

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def _prepare_batch(self, batch):
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            elif isinstance(value, (list, str)):
                prepared_batch[key] = value
            else:
                try:
                    prepared_batch[key] = torch.tensor(value).to(self.device)
                except:
                    prepared_batch[key] = value
        return prepared_batch

    def training_step(self, batch):
        try:
            batch = self._prepare_batch(batch)
            
            required_keys = ['question', 'image', 'answer']
            missing_keys = [key for key in required_keys if key not in batch]
            if missing_keys:
                raise KeyError(f"Batch is missing required keys: {missing_keys}")
            
            try:
                logits, loss = self.base_model(
                    batch['question'],
                    batch['image'],
                    batch['answer']
                )
            except RuntimeError as e:
                print(f"Error in forward pass: {str(e)}")
                print(f"Batch shapes - Question: {batch['question'].shape if isinstance(batch['question'], torch.Tensor) else 'not tensor'}, "
                      f"Image: {batch['image'].shape if isinstance(batch['image'], torch.Tensor) else 'not tensor'}, "
                      f"Answer: {batch['answer'].shape if isinstance(batch['answer'], torch.Tensor) else 'not tensor'}")
                raise
                
            return loss
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise

    def validation_step(self, batch):
        with torch.no_grad():
            batch = self._prepare_batch(batch)
            loss = self.base_model(
                batch['question'],
                batch['image'],
                batch['answer']
            )[1]
            pred_answers = self.base_model(batch['question'], batch['image'])
            return loss, pred_answers, batch['answer']

    def training(self, train_loader, valid_loader):
        """Main training loop"""
        print("Starting training...")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        initial_epoch, best_score = self.load_checkpoint(
            os.path.join(self.save_path, 'last_model.pth')
        )

        threshold = 0
        self.base_model.train()

        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            train_loss = 0.0
            valid_loss = 0.0
            
            # Training loop
            self.base_model.train()
            with tqdm(desc=f'Training', total=len(train_loader)) as pbar:
                for it, batch in enumerate(train_loader):
                    try:
                        self.optimizer.zero_grad()
                        
                        with torch.cuda.amp.autocast():
                            loss = self.training_step(batch)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        train_loss += loss.item()
                        pbar.set_postfix({
                            'loss': f"{train_loss/(it+1):.4f}",
                            'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                        })
                        pbar.update()
                    except Exception as e:
                        print(f"Error in training iteration {it}: {str(e)}")
                        continue

            self.scheduler.step()
            train_loss /= len(train_loader)

            # Validation loop
            self.base_model.eval()
            metrics = {'wups': 0.0, 'em': 0.0, 'f1': 0.0, 'cider': 0.0}
            valid_batches = 0
            
            with tqdm(desc=f'Validation', total=len(valid_loader)) as pbar:
                for batch in valid_loader:
                    try:
                        loss, preds, targets = self.validation_step(batch)
                        valid_loss += loss.item()
                        
                        # Update metrics
                        metrics['wups'] += self.compute_score.wup(targets, preds)
                        metrics['em'] += self.compute_score.em(targets, preds) 
                        metrics['f1'] += self.compute_score.f1_token(targets, preds)
                        metrics['cider'] += self.compute_score.cider_score(targets, preds)
                        
                        valid_batches += 1
                        pbar.update()
                    except Exception as e:
                        print(f"Error in validation step: {str(e)}")
                        continue

            # Calculate average metrics
            valid_loss /= valid_batches
            metrics = {k: v/valid_batches for k,v in metrics.items()}
            
            # Log metrics
            log_str = (f"Epoch {epoch + 1}/{self.num_epochs + initial_epoch} - "
                      f"Train loss: {train_loss:.4f} - "
                      f"Valid loss: {valid_loss:.4f} - "
                      f"Metrics: {metrics}")
            print(log_str)
            
            with open(os.path.join(self.save_path, 'training_log.txt'), 'a') as f:
                f.write(log_str + '\n')
            
            # Save checkpoint
            score = metrics[self.best_metric]
            self.save_checkpoint(epoch, score, is_best=score > best_score)

            if score > best_score:
                best_score = score
                print(f"New best model with {self.best_metric} of {score:.4f}")
                threshold = 0
            else:
                threshold += 1
                print(f"No improvement in {threshold} epochs")

            if threshold >= self.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        print("Training completed!")
        return best_score

    def log_metrics(self, epoch, train_loss, valid_loss, metrics):
        """Log training metrics to file and console"""
        log_str = (f"Epoch {epoch + 1} - "
                  f"Training loss: {train_loss:.4f} - "
                  f"Valid loss: {valid_loss:.4f} - "
                  f"Valid metrics: {metrics}")
        
        print(log_str)
        with open(os.path.join(self.save_path, 'log.txt'), 'a') as f:
            f.write(log_str + '\n')
