from common_imports import *

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
    Dataset class for the ViVQA dataset with improved error handling and index management
    """
    def __init__(self, df, img_dir):
        """
        Initialize dataset
        Args:
            df: pandas DataFrame containing the dataset
            img_dir: directory containing images
        """
        # Reset index để đảm bảo index liên tục từ 0
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.length = len(self.df)
        
        # Validate image directory
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
            
    def __len__(self):
        """Return the total number of samples"""
        return self.length
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        Args:
            idx: index of the sample
        Returns:
            dict: Sample containing image path, question and answer
        """
        # Validate index
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.length}")
            
        try:
            # Get row using iloc instead of loc
            row = self.df.iloc[idx]
            
            # Extract data
            question = str(row['question'])
            answer = str(row['answer'])
            image_id = str(row['img_id'])
            
            # Construct image path
            img_file = os.path.join(self.img_dir, f'image_{image_id}.jpg')
            
            # Validate image file exists
            if not os.path.exists(img_file):
                print(f"Warning: Image file not found: {img_file}")
                img_file = None  # or provide a default image path
                
            return {
                'image': img_file,
                'question': question,
                'answer': answer
            }
            
        except Exception as e:
            print(f"Error accessing sample at index {idx}: {str(e)}")
            # Return a default sample or raise the exception
            raise e
            
    def get_sample_info(self, idx):
        """
        Get detailed information about a sample for debugging
        Args:
            idx: index of the sample
        Returns:
            dict: Detailed information about the sample
        """
        if idx < 0 or idx >= self.length:
            return f"Index {idx} out of bounds for dataset with length {self.length}"
            
        try:
            row = self.df.iloc[idx]
            return {
                'index': idx,
                'dataframe_index': row.name,
                'question': row['question'],
                'answer': row['answer'],
                'img_id': row['img_id'],
                'type': row['type'] if 'type' in row else None
            }
        except Exception as e:
            return f"Error getting sample info: {str(e)}"

def create_datasets(train_df, valid_df, test_df, images_dir):
    """
    Create train, validation and test datasets with validation
    Args:
        train_df: training DataFrame
        valid_df: validation DataFrame
        test_df: testing DataFrame
        images_dir: directory containing images
    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    # Validate image directory
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
        
    # Validate DataFrames
    for name, df in [("Training", train_df), ("Validation", valid_df), ("Test", test_df)]:
        if df is None or len(df) == 0:
            raise ValueError(f"{name} DataFrame is empty")
        required_columns = ['question', 'answer', 'img_id']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} DataFrame missing required columns: {missing_cols}")
            
    # Create datasets
    train_dataset = ViVQA_Dataset(train_df, images_dir)
    valid_dataset = ViVQA_Dataset(valid_df, images_dir)
    test_dataset = ViVQA_Dataset(test_df, images_dir)
    
    # Print dataset information
    print(f'[INFO] Train size: {len(train_dataset)}')
    print(f'[INFO] Valid size: {len(valid_dataset)}')
    print(f'[INFO] Test size:  {len(test_dataset)}')
    
    return train_dataset, valid_dataset, test_dataset
