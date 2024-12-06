import torch
import torchvision.transforms as transforms
from PIL import Image
import torch_xla.core.xla_model as xm


def load_and_clean_data(train_path, test_path):
    """
    Load and clean training and testing data
    Args:
        train_path: path to train.csv
        test_path: path to test.csv
    Returns:
        train_df, test_df, valid_df: cleaned DataFrames
    """
    # Load train data
    train_df = pd.read_csv(train_path)
    if train_df.shape[1] > 4:
        train_df = train_df.iloc[:, 1:]  # Drop first column (index)
        
    # Load test data
    test_df = pd.read_csv(test_path)
    if test_df.shape[1] > 4:
        test_df = test_df.iloc[:, 1:]  # Drop first column (index)

    # Prepare data for train-validation split
    X = train_df[['question', 'answer', 'img_id', 'type']]

    # Add a dummy target variable
    train_df['dummy_target'] = train_df['type']

    # Split into train and validation sets
    train_X, valid_X, train_dummy, valid_dummy = train_test_split(
        X, 
        train_df['dummy_target'], 
        test_size=0.2, 
        random_state=42, 
        stratify=train_df['dummy_target']
    )

    # Create dataframes for training and validation sets
    train_df = pd.DataFrame({
        'question': train_X['question'], 
        'answer': train_X['answer'], 
        'img_id': train_X['img_id'], 
        'type': train_X['type']
    })
    valid_df = pd.DataFrame({
        'question': valid_X['question'], 
        'answer': valid_X['answer'], 
        'img_id': valid_X['img_id'], 
        'type': valid_X['type']
    })

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
        
    print(f'Train shape: {train_df.shape}')
    print(f'Test shape: {test_df.shape}')
    print(f'Valid shape: {valid_df.shape}')
    
    return train_df, test_df, valid_df

def remove_duplicates(df, save_path=None):
    """
    Remove duplicates from DataFrame and optionally save to csv
    Args:
        df: input DataFrame
        save_path: path to save cleaned DataFrame (optional)
    Returns:
        cleaned DataFrame
    """
    df.drop_duplicates(keep=False, inplace=True)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

class ViVQA_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the ViVQA dataset.
    """
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        # Nếu không có transform, sử dụng transform mặc định
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        question = self.df.loc[idx, 'question']
        answer = self.df.loc[idx, 'answer']
        image_id = self.df.loc[idx, 'img_id']
        quest_type = self.df.loc[idx, 'type']
        img_file = os.path.join(self.img_dir, f'image_{image_id}.jpg')
        # Mở và transform image
        image = Image.open(img_file).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'question': question, 'answer': answer}

def create_datasets(train_df, valid_df, test_df, images_dir):
    """
    Create train, validation and test datasets
    Args:
        train_df: training DataFrame
        valid_df: validation DataFrame
        test_df: testing DataFrame
        images_dir: directory containing images
    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    train_dataset = ViVQA_Dataset(train_df, images_dir)
    valid_dataset = ViVQA_Dataset(valid_df, images_dir)
    
    test_df.reset_index(drop=True, inplace=True)
    test_dataset = ViVQA_Dataset(test_df, images_dir)
    
    print(f'[INFO] Train size: {len(train_dataset)}')
    print(f'[INFO] Valid size: {len(valid_dataset)}')
    print(f'[INFO] Test size:  {len(test_dataset)}')
    
    return train_dataset, valid_dataset, test_dataset
