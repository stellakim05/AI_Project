#These statements import the libraries needed to run the BERT-based classifier
import pandas as pd #CSV file read, Data frame manipulation
import torch #Deep Learning Framework
import torch.nn as nn #Neural network module and layer class 
from torch.utils.data import Dataset, DataLoader #Data batch processing
from transformers import BertTokenizer, BertForSequenceClassification #Using pre-trained BERT models, tokenizers
from torch.optim import AdamW #Learning model parameters
from sklearn.model_selection import train_test_split, StratifiedKFold #Data segmentation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix #Performance metrics calculation
import numpy as np #Calculate statistics 
import matplotlib.pyplot as plt #Generating graphs 
import seaborn as sns #Visualizing statistical data
from tqdm import tqdm #Progress bar 
import os #File/directory operations

#Defines the hyperparameters required for model training and evaluation
MODEL_NAME = "bert-base-uncased" #BERT model name in lower case
MAX_LENGTH = 256 #Maximum number of tokens when tokenizing text
BATCH_SIZE = 32  #Number of samples to process at a time
LEARNING_RATE = 2e-5 #Learning rate (2 × 10^-5 = 0.00002)
NUM_EPOCHS = 2 #Epoch (how many times to learn the entire dataset)
RANDOM_SEED = 42 #Random seed (fixed for reproducibility)
N_FOLDS = 5  #Number of folds used in Cross-validation
SAVE_MODEL = True  #Save the model after completion of training
MODEL_SAVE_PATH = "bert_classification_model"  #Directory path to store model

#Save the three paper abstract CSV file paths as a list
CSV_FILES = [
    "abstracts_bio_info.csv", #Bioinformatics dataset
    "abstracts_neuro.csv", #Neuroscience  dataset
    "abstracts_environ.csv" #Environmental Science dataset
]

#Configuring devices for PyTorch to Use
#If GPU(CUDA) is available, use GPU (else, use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Print out the current device 
print(f"Using device: {device}")


#Data loading and preprocessing
def load_data():
    """
    This function reads and combines the CSV files of three 
    scientific disciplines into one. Then, removes rows with 
    missing values from the columns 'Abstract' and Label'. 
    """
    all_data = []
    
    #Repeat for each files in the CSV_Files list 
    for csv_file in CSV_FILES:
        #Function to check if the file exists
        if os.path.exists(csv_file):
            #Read CSV file and convert it to pandas DataFrame
            df = pd.read_csv(csv_file)
            #Returns the number of rows of DataFrame
            print(f"Loaded {len(df)} samples from {csv_file}")
            #Add DataFrame to List
            all_data.append(df)
        else:
            #If the file does not exist, print a warning message and move on to the next file
            print(f"Warning: {csv_file} not found. Skipping...")
    #If the list is empty, cause an error
    if not all_data:
        raise FileNotFoundError("No CSV files found!")
    
    #Combine multiple DataFrames into one 
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #Remove rows with missing value (NaN)
    combined_df = combined_df.dropna(subset=["Abstract", "Label"])
    
    #Shows the samples
    #Count the number of values in each column and return
    print(f"\nTotal samples: {len(combined_df)}")
    print(f"Label distribution:\n{combined_df['Label'].value_counts()}")
    
    return combined_df


def prepare_labels(df):
    """
    Since the model cannot handle text labels directly, it converts each label
    to a number (0, 1, 2). It generates a bidirectional mapping dictionary to 
    convert the prediction results back to the original label.
    """

    #Extract unique label values 
    labels = df["Label"].unique() 
    #sorted(): sorted alphabetically, enumerate(): create index and value pairs
    #Create dictionary: {label: index} 
    label_to_id = {label: idx for idx, label in enumerate(sorted(labels))} 
    #Create dictionary: {label: index} (reverse mapping)
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    #Create a new 'label_id' column by converting labels to numeric IDs
    df["label_id"] = df["Label"].map(label_to_id)
    
    print(f"\nLabel mapping: {label_to_id}")
    
    #Return multiple values in a tuple
    return df, label_to_id, id_to_label


#Dataset class definition
class AbstractDataset(Dataset):
    """
    This class inherits PyTorch's Dataset class, so the paper
    abstract data can be used. It converts to a format that the
    BERT model can handle (input_ids, attention_mask) and used 
    with DataLoader to provide data in batches during training
    and evaluation. 
    """
    
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initialize the dataset. 

        Args:
            texts: array of abstract text strings
            labels: array of numeric label IDs
            tokenizer: BERT tokenizer
            max_length: maximum number of tokens (256)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Returns the number of samples in the dataset. 

        This method enables the use of len() function, 
        and it needs for PyTorch DataLoader to know the 
        size of the dataset. 

        Returns: 
            int: number of text-label pairs in the dataset 
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Take a sample of data to the index and tokenize it. 

        This method enables indexing, and calls automatically 
        that PyTorch DataLoader can create a batch. 

        Args:
            idx (int): index of samples to import (starting from 0)

        Returns:
            dict: dictionary including input_ids, attention_mask, labels    
        """
        #Converting text to string (handling different data types)
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Convert text to numeric vectors using tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            #encoding["input_ids"]: 2d tensor that has shape [1, max_length] 
            #.flatten(): flatten to [max_length] by removing batch dimension
            "input_ids": encoding["input_ids"].flatten(),
            #encoding["attention_mask"]: mark the actual token as 1 and the padding as 0
            "attention_mask": encoding["attention_mask"].flatten(),
            #torch.tensor(): convert Python integers to PyTorch tensors
            #dtype=torch.long: integer type (classification label must be integer)
            "labels": torch.tensor(label, dtype=torch.long)
        }

#Model training function
def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    """
    This function implements the training loop: forward propagation, loss calculation, 
    backpropagation, and parameter update. It also performs validation for each 
    epoch and monitor model performance during training. 

    Args: 
        model: BERT model instance to learn
        train_loader: DataLoader that provides training batches 
        val_loader: Dataloader that provides validation data placement
        num_epochs: Number of times the entire dataset is learned
        Learning_rate: Optimizer's Learning Rate

    Returns:
        tuple: (Learning loss list, verification accuracy list) 
    """
    #AdamW(): Create AdamW Optimizer 
    #model.parameters(): Returns all learnable parameters of the model
    #lr=learning_rate: setting the learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    #range(): Generates a numeric sequence from 0 to num_epochs-1
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        #Training mode
        #model.train(): Set model to learning mode
        model.train()
        total_train_loss = 0
        
        #tqdm(): Create progress bar (visual learning progress)
        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:

            #.to(device): Move the tensor to the GPU (if possible) or CPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            #optimizer.zero_grad(): Initialize gradients from previous batches 
            optimizer.zero_grad()
            
            #Forward pass: Pass inputs to the model to calculate output and loss
            #When provide labels, the model automatically calculates the loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            #outputs.loss: Cross entropy loss for this batch 
            loss = outputs.loss

            #.item(): Extract scalar value from tensor
            #Add batch loss to total loss
            total_train_loss += loss.item()
    
            #loss.backward(): Perform backpropagation 
            loss.backward()

            #optimizer.step(): Update model parameters using calculated gradients
            optimizer.step()
            
            #.set_postfix(): Show lost value of current batch in progress bar
            train_progress.set_postfix({"loss": loss.item()})
        
        #Calculate average learning loss
        avg_train_loss = total_train_loss / len(train_loader)
        #list.append(): Add average loss to list
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        #Validation
        #evaluate_model(): Evaluate the model from validation data
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_accuracies


def evaluate_model(model, data_loader):
    """
    This function sets the model to evaluate mode, performs
    predictions on all samples of the data loader, and then calculates 
    its accuracy. Because gradients do not calculate, it saves memory and speed up inference. 

    Args:
        model: learned BERT model to evaluate
        data_loader: Dataloader that provides placement of evaluation batches

    Returns:
        float: Accuracy score (0.0 to 1.0, correct prediction ratio)
    """

    #model.eval(): set model to evaluation mode
    model.eval()
    predictions = []
    true_labels = []
    
    #torch.no _grad(): Disable gradient calculation 
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            #Forward pass (without labels): only perform predictions, no losses calculated
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            #outputs.logits: prediction score (logit), shape [batch_size, num_classes] for each class
            logits = outputs.logits

            #torch.argmax(): Find the class index with the highest logit for each sample
            #dim=1: Find maximum value in class dimension
            preds = torch.argmax(logits, dim=1)
            
            #.cpu(): move tensor from GPU to CPU (required before NumPy conversion)
            #.numpy(): Convert PyTorch tensors to NumPy arrays
            #list.extend(): Add all elements of the array to the list (adding by element, unlike list.append())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    #Accuracy_score(): Calculate accuracy (correct prediction ratio)
    #Returns accuracy by comparing true_labels with predictions
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

#Performance evaluation and visualization
def detailed_evaluation(model, test_loader, id_to_label):
    """
    The function evaluates the model from test data and 
    computes a evaluation metrics: Accuracy per class and 
    overall (macro-average), Precision, Recall, and F1-score. 
    It also generates and visualizes confusion matrix heatmap. 
    
    Args:
        model: learned BERT model to evaluate
        test_loader: Dataloader to provide test data placement
        id_to_label: a dictionary that converts numeric IDs into label strings

    Returns:
        tuple: (accuracy, precision array, recall array, F1 array, confusion matrix)
    """
    model.eval()
    predictions = []
    true_labels = []
    
    #tqdm(): Assessment progress bar
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    #Calculate performance metrics
    #accuracy_score(): Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predictions)

    #precision_recall_fscore_support(): calculate precision, recall, F1-score for each class
    #average=None: Return individual indicators by class
    #zero_division=0: 0 return when an error occurs dividing by 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    #Overall average(macro average: equal weight for each class)
    #np.mean(): Calculate the mean for all classes
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print("\n" + "=" * 50)
    print("Performance Evaluation Results")
    print("=" * 50)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Overall Macro Precision: {macro_precision:.4f}")
    print(f"Overall Macro Recall: {macro_recall:.4f}")
    print(f"Overall Macro F1-score: {macro_f1:.4f}")
    
    print("\nPerformance by field:")
    print("-" * 50)

    #List comprehension: Create a list of label names in order of ID
    #range(len(id_to_label)) : Create indexes such as 0, 1, 2
    labels_list = [id_to_label[i] for i in range(len(id_to_label))]

    #enumerate(): Repeat the index and value together
    for i, label in enumerate(labels_list):
        print(f"{label}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")
    
    #Create Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    #Confusion matrix visualization
    plt.figure(figsize=(10, 8))
    # sns.heatmap(): Visualize the confusion matrix as a heatmap
    # - cm: Confusion matrix data (2D array)
    # - annot=True: Display numeric values in each cell
    # - fmt="d": in integer format
    # - cmap="Blues": color map (dark blue = high value)
    # - xticklabels/yticklabels: axis label (class name)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_list,
        yticklabels=labels_list
    )

    #plt.title(): setting graph title
    plt.title("Confusion Matrix")

    #plt.ylabel(): y-axis label setting (actual label)
    plt.ylabel("Actual Label")

    #plt.xlabel(): x-axis label setting (predicted label)
    plt.xlabel("Predicted Label")

    #plt.tight_layout(): adjust subplot parameters 
    plt.tight_layout()

    #plt.show(): graph display
    plt.show()
    
    return accuracy, precision, recall, f1, cm


def cross_validate(X, y, tokenizer, label_to_id, id_to_label, n_folds=5):
    """
    This function divides the data into k folds, learns with k-1 folds for 
    each fold, validates with the remaining 1 fold, and repeats for all folds.
    Then, it calculates the mean performance and standard deviation of k folds.
    
    Args:
        X: array of abstract texts 
        y: numeric label ID array 
        tokenizer: BERT tokenizer
        label_to_id: a dictionary that converts labels into IDs
        id_to_label: a dictionary that converts an ID to a label
        n_folds: number of cross-validation folds (default: 5)

    Returns:
        tuple: Average indicators for all folds (accuracy, precision, recall, F1-score)
    """
    print("\n" + "=" * 50)
    print(f"Performing {n_folds}-Fold Cross-Validation")
    print("=" * 50)
    
    #StratifiedKFold(): k-fold cross-validation generator maintaining class ratio
    #n_splits=n_folds: Number of folds (5)
    #shuffle=True: Mixing data randomly before splitting
    #random_state=RANDOM_SEED: Fixed random seeds for reproducibility
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    #enumerate(skf.split(X, y),1): repeat fold index and (learning index, verification index) tuple
    #skf.split(): an iterator that generates a learning/verification index for each fold
    # enumerate(..., 1): start index from 1 (1 instead of 0)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*50}")
        
        #Use index arrays to select the corresponding rows
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        print(f"Train samples: {len(X_train_fold)}, Val samples: {len(X_val_fold)}")
        
        # Create datasets
        train_dataset = AbstractDataset(X_train_fold, y_train_fold, tokenizer, MAX_LENGTH)
        val_dataset = AbstractDataset(X_val_fold, y_val_fold, tokenizer, MAX_LENGTH)
        
        #DataLoader(): Create loaders that provide data in batches
        #batch_size=BATCH_SIZE: batch size (32)
        #shuffle=True: Learning data is randomly mixed at every epoch
        #shuffle=False: validation data is not mixed (no need)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        #Load model for this fold
        num_labels = len(label_to_id)
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels
        )
        model.to(device)
        
        #Train model
        train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
        
        #Evaluate
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        #list.append(): add indicators for this fold to the list
        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
    
    #Print cross-validation summary
    print("\n" + "=" * 50)
    print("Cross-Validation Summary")
    print("=" * 50)

    #np.mean(): average calculation of all folds
    #np.std(): calculate the standard deviation of all folds
    #Output in the form of mean ± standard deviation
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean Precision: {np.mean(fold_precisions):.4f} (+/- {np.std(fold_precisions):.4f})")
    print(f"Mean Recall: {np.mean(fold_recalls):.4f} (+/- {np.std(fold_recalls):.4f})")
    print(f"Mean F1-score: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
    
    #Return the mean indicator for all folds
    return np.mean(fold_accuracies), np.mean(fold_precisions), np.mean(fold_recalls), np.mean(fold_f1s)


def main():
    """
    This function adjusts all steps: data loading, preprocessing, model initialization, 
    cross-validation, final model learning, evaluation, and model storage. 

    Steps: 
    1. Load and combine data from CSV files
    2. Prepare labels (text to numeric conversion)
    3. Split data into training/test sets
    4. BERT tokenizer load 
    5. Perform cross-validation
    6. Learning the final model with the full training set
    7. Evaluate in a test set
    8. Save Model (if SAVE_MODEL is True) 
    """
    print("=" * 50)
    print("BERT Paper Abstract Classification Model")
    print("=" * 50)
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    df = load_data()
    
    #Prepare labels
    df, label_to_id, id_to_label = prepare_labels(df)
    
    #2. Split data for cross-validation and final test
    print("\n[Step 2] Splitting data...")
    texts = df["Abstract"].values
    labels = df["label_id"].values
    
    #Train/Test split (80:20) - test set is held out for final evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"Full training set for CV: {len(X_train_full)}")
    print(f"Test set (held out): {len(X_test)}")
    
    #3. Load tokenizer
    print("\n[Step 3] Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    num_labels = len(label_to_id)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of classes: {num_labels}")
    
    #4. Perform cross-validation
    print(f"\n[Step 4] Performing {N_FOLDS}-fold cross-validation...")
    cv_acc, cv_prec, cv_rec, cv_f1 = cross_validate(
        X_train_full, y_train_full, tokenizer, label_to_id, id_to_label, N_FOLDS
    )
    
    #5. Train final model on full training set
    print("\n[Step 5] Training final model on full training set...")
    #Split full training set into train/val for final model training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    print(f"Final train samples: {len(X_train)}")
    print(f"Final validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    #Load model for final training
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    model.to(device)
    
    #Create datasets and data loaders
    train_dataset = AbstractDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = AbstractDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = AbstractDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #Train final model
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE
    )
    
    #6. Final evaluation on test data
    print("\n[Step 6] Final performance evaluation...")
    accuracy, precision, recall, f1, cm = detailed_evaluation(
        model, test_loader, id_to_label
    )
    
    #7. Save the trained model
    if SAVE_MODEL:
        print(f"\n[Step 7] Saving model to {MODEL_SAVE_PATH}...")
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        #Save model and tokenizer
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        #Save label mappings
        import json
        with open(f"{MODEL_SAVE_PATH}/label_mappings.json", "w") as f:
            json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)
        
        print(f"Model saved successfully to {MODEL_SAVE_PATH}/")
        print("To load the model later:")
        print(f"  model = BertForSequenceClassification.from_pretrained('{MODEL_SAVE_PATH}')")
        print(f"  tokenizer = BertTokenizer.from_pretrained('{MODEL_SAVE_PATH}')")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)

#Run the main() function only when the script runs directly
if __name__ == "__main__":
    main()