# trainer/vqa_task.py
from common_imports import *
from models.vqa_model import MBart_BEiT_Model
from models.encoders import Bart_Encode_Feature, Vision_Encode_Pixel
from metrics.ScoreCalculator import ScoreCalculator
from models.Bart_Encode_Feature import Bart_tokenizer

class VQA_Task:
    def __init__(self, config):
        self.save_path = os.path.join(config.TRAINING.SAVE_PATH, config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = config.TRAINING.EPOCHS
        self.patience = config.TRAINING.PATIENCE
        self.best_metric= config.TRAINING.METRIC_BEST
        self.learning_rate = config.TRAINING.LEARNING_RATE
        self.weight_decay = config.TRAINING.WEIGHT_DECAY
        self.batch_size = config.TRAINING.BATCH_SIZE
        self.num_workers = config.TRAINING.WORKERS

        self.tokenizer=Bart_tokenizer(config.MODEL)
        self.base_model= MBart_BEiT_Model(config.MODEL).to(self.device)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.9 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.compute_score = ScoreCalculator()

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            print('')
            valid_loss = 0.
            train_loss = 0.
            # Train
            with tqdm(desc=f'Epoch {epoch + 1} - Training  ', unit='it', total=len(train)) as pbar:
              for it, item in enumerate(train):
                  logits, loss = self.base_model(item['question'], item['image'], item['answer'])
                  self.scaler.scale(loss).backward()
                  self.scaler.step(self.optimizer)
                  self.scaler.update()
                  self.optimizer.zero_grad()
                  train_loss += loss
                  cur_loss = train_loss.detach().cpu().numpy() / (it + 1)
                  pbar.set_postfix({'loss': f"{cur_loss:.4f}", 'lr':self.scheduler.get_last_lr()[0]})
                  pbar.update()
            self.scheduler.step()
            train_loss /=len(train)
            # Valid
            valid_em = 0.
            valid_wups=0.
            valid_f1 =0.
            valid_cider=0.
            with torch.no_grad():
              with tqdm(desc=f'Epoch {epoch + 1} - Evaluation', unit='it', total=len(valid)) as pbar:
                for it, item in enumerate(valid):
                    valid_loss += self.base_model(item['question'], item['image'], item['answer'])[1]
                    pred_answers = self.base_model(item['question'],item['image'])
                    clean_answers=item['answer']
                    valid_wups +=self.compute_score.wup(clean_answers,pred_answers)
                    valid_em +=self.compute_score.em(clean_answers,pred_answers)
                    valid_f1 +=self.compute_score.f1_token(clean_answers,pred_answers)
                    valid_cider +=self.compute_score.cider_score(clean_answers,pred_answers)
                    pbar.update()
            valid_loss /=len(valid)
            valid_wups /= len(valid)
            valid_em /= len(valid)
            valid_f1 /= len(valid)
            valid_cider/=len(valid)
            print(f"Training loss: {train_loss:.4f} - Valid loss: {valid_loss:.4f} - Valid wups: {valid_wups:.4f} - Valid em: {valid_em:.4f} - Valid f1: {valid_f1:.4f} - Valid cider: {valid_cider:.4f}")
            with open(os.path.join(self.save_path, 'log.txt'), 'a') as file:
                file.write(f"Epoch {epoch + 1} - Training loss: {train_loss:.4f} - Valid loss: {valid_loss:.4f} - Valid wups: {valid_wups:.4f} - Valid em: {valid_em:.4f} - Valid f1: {valid_f1:.4f} - Valid cider: {valid_cider:.4f}\n")

            if self.best_metric =='em':
                score=valid_em
            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='wups':
                score=valid_wups
            if self.best_metric=='cider':
                score=valid_cider

            # Save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))

            # Save the best model
            if epoch > 0 and score <= best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"Saved the best model with {self.best_metric} of {score:.4f}")

            # early stopping
            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    def get_predict(self):
      # Load the model
      print("Loadding best model...")
      if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
          checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
          self.base_model.load_state_dict(checkpoint['model_state_dict'])
      else:
          print("Prediction require the model must be trained. There is no weights to load for model prediction!")
          raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")
      # Create test_loader
      if test_dataset:
          print(f"[INFO] Test size: {len(test_dataset)}")
          test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
      else:
          raise Exception("Not found test dataset")

      print("Predicting...")
      img_path=[]
      quests=[]
      gts=[]
      preds=[]
      self.base_model.eval()
      with torch.no_grad():
          for it,item in enumerate(tqdm(test_loader)):
              answers = self.base_model(item['question'],item['image'])
              preds.extend(answers)
              img_path.extend(item['image'])
              quests.extend(item['question'])
              gts.extend(item['answer'])
      test_wups=self.compute_score.wup(gts,preds)
      test_em=self.compute_score.em(gts,preds)
      test_f1=self.compute_score.f1_token(gts,preds)
      test_cider=self.compute_score.cider_score(gts,preds)
      print(f"\nEvaluation scores on Test - wups: {test_wups:.4f} - em: {test_em:.4f} - f1: {test_f1:.4f} - cider: {test_cider:.4f}")

      data = {
          "img_path":img_path,
          "question": quests,
          "ground_truth":gts,
          "predicts": preds
      }
      df = pd.DataFrame(data)
      df.to_csv(os.path.join(self.save_path,'result.csv'), index=False)
      print(f"Save result to: {os.path.join(self.save_path,'result.csv')}")

