import os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize,
                                    ToTensor)
from transformers import ViTForImageClassification,ViTImageProcessor
import accelerate
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import seaborn as sns
from transformers import set_seed

# Dependencies and data loading


DATA_DIR = 'D:/SE_AI/Sem_4/Deep_Learning/Group_Project/data/processed'

torch.manual_seed(42)


# Define paths
train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'test')

# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# Load and label the training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Load and label the validation dataset
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Load and label the test dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


images, labels = next(train_generator)
print("Sample batch shape from train_generator:", images.shape)

# Preprocessing

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
print("ViT Input Size: ", size)

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            Resize((size, size)),
            RandomRotation(15),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Apply transforms to the train and validation datasets
def apply_transforms(generator, transform_fn, process_all=False):
    transformed_images = []
    transformed_labels = []
    num_batches = len(generator) if process_all else 1
    print(f"Processing {num_batches} batch(es) from generator...")
    for i in range(num_batches):
        try:
            images, labels = next(generator)
            for j in range(len(images)):
                image = Image.fromarray((images[j] * 255).astype('uint8'))  # Convert to PIL Image
                transformed_image = transform_fn({'image': [image]})['pixel_values'][0]
                transformed_images.append(transformed_image)
                transformed_labels.append(labels[j])
        except StopIteration:
            print("Reached end of generator.")
            break # Exit loop if generator is exhausted
    print(f"Processed {len(transformed_images)} images.")
    return np.stack(transformed_images), np.array(transformed_labels)


PROCESS_ALL_DATA = False 

print("Applying train transforms...")
train_images, train_labels = apply_transforms(train_generator, train_transforms, process_all=PROCESS_ALL_DATA)
print("Applying validation transforms...")
val_images, val_labels = apply_transforms(val_generator, val_transforms, process_all=PROCESS_ALL_DATA)
print("Applying test transforms...")
test_images, test_labels = apply_transforms(test_generator, val_transforms, process_all=PROCESS_ALL_DATA)


# Define label2id mapping
label2id = {0.0: 0, 1.0: 1}
id2label = {0: "no_tumor", 1: "tumor"}


unique, counts = np.unique(train_labels, return_counts=True)

# Plot the distribution
plt.figure() # Create a new figure
plt.bar([id2label[int(label)] for label in unique], counts, color=['blue', 'orange']) # Ensure labels are int keys
for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

plt.xlabel('Labels')
plt.ylabel('Number of Images (in loaded batch/data)')
plt.title('Loaded Training Data Distribution')
# plt.show()

# Convert train and validation data into a format compatible with DataLoader
train_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(train_images, train_labels)]
val_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(val_images, val_labels)]
test_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(test_images, test_labels)]


# Define collate function
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # Ensure labels are correctly mapped using integer keys
    labels = torch.tensor([label2id[float(example["label"])] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Create DataLoaders
train_dataloader = DataLoader(train_data, collate_fn=collate_fn, batch_size=4)
val_dataloader = DataLoader(val_data, collate_fn=collate_fn, batch_size=4)
test_dataloader = DataLoader(test_data, collate_fn=collate_fn, batch_size=4) # Create test dataloader too


print("Checking train dataloader batch shapes:")
try:
    batch = next(iter(train_dataloader))
    for k,v in batch.items():
      if isinstance(v, torch.Tensor):
        print(k, v.shape)
except StopIteration:
    print("Train dataloader is empty (likely processed 0 images).")


print("\nChecking validation dataloader batch shapes:")
try:
    batch = next(iter(val_dataloader))
    for k,v in batch.items():
      if isinstance(v, torch.Tensor):
        print(k, v.shape)
except StopIteration:
    print("Validation dataloader is empty (likely processed 0 images).")


# Display some images from the loaded training data
if len(train_images) > 0:
    plt.figure() # Create new figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    num_images_to_show = min(len(train_images), 9)
    for i in range(num_images_to_show):
        ax = axes[i]
        img_display = train_images[i].transpose(1, 2, 0) # C, H, W -> H, W, C
        mean = np.array(image_mean)
        std = np.array(image_std)
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        ax.imshow(img_display)
        ax.set_title(f"Label: {id2label[int(train_labels[i])]}") # Ensure label is int key
        ax.axis('off')
    # Hide unused subplots
    for i in range(num_images_to_show, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No training images loaded to display.")

# Model loading + fine tuning

# Load the model with the corrected mappings
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  ignore_mismatched_sizes=True)


metric_name = "accuracy"

args = TrainingArguments(
    "checkpoints",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,  # Increased learning rate
    per_device_train_batch_size=8,  # Reduced batch size for better gradient updates from 32 to 16 to 8
    per_device_eval_batch_size=8,  # Increased evaluation batch size from 4 to 8
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    logging_steps=10, # Log more frequently
    remove_unused_columns=False,
    report_to="tensorboard",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Set the random seed for reproducibility
set_seed(62)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)

# Running and model evaluation

if len(train_data) > 0 and len(val_data) > 0:
    print("\nStarting Training...")
    trainer.train()

    print("\nTraining Log History:")
    print(trainer.state.log_history)

    # Extract training and validation accuracy from trainer's logs
    train_logs = trainer.state.log_history
    eval_accuracy = [log["eval_accuracy"] for log in train_logs if "eval_accuracy" in log]
    eval_loss = [log["eval_loss"] for log in train_logs if "eval_loss" in log]
    eval_precision = [log["eval_precision"] for log in train_logs if "eval_precision" in log]
    eval_recall = [log["eval_recall"] for log in train_logs if "eval_recall" in log]
    eval_f1 = [log["eval_f1"] for log in train_logs if "eval_f1" in log]
    epochs = [log["epoch"] for log in train_logs if "eval_loss" in log] # Get epochs for plotting

    # Plotting eval accuracy and eval loss
    if eval_accuracy and eval_loss:
        plt.figure() # Create new figure
        fig, ax1 = plt.subplots()

        color = 'blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(epochs, eval_accuracy, color=color, label='Accuracy', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'red'
        ax2.set_ylabel('Loss', color=color)
        ax2.plot(epochs, eval_loss, color=color, label='Loss', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # to prevent overlap of labels
        plt.title('Validation Accuracy and Loss')
        plt.show()
        plt.savefig('validation_accuracy_loss.png') # Save the plot
        print("Saved validation_accuracy_loss.png")
    else:
        print("Not enough data points to plot validation accuracy/loss.")


    # Plotting precision, recall, and F1 score
    if eval_precision and eval_recall and eval_f1:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, eval_precision, label='Precision', marker='o', color='blue')
        plt.plot(epochs, eval_recall, label='Recall', marker='o', color='green')
        plt.plot(epochs, eval_f1, label='F1 Score', marker='o', color='red')

        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Validation Precision, Recall, and F1 Score Over Epochs')
        plt.legend()
        plt.grid(True)
        # plt.show() # Display plot - uncomment or save figure
        plt.savefig('validation_prf1_scores.png') # Save the plot
        print("Saved validation_prf1_scores.png")
    else:
        print("Not enough data points to plot validation P/R/F1 scores.")


    # Save the trained model and processor
    print("\nSaving model and processor...")
    save_directory = "saved_model"
    model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)
    print(f"Model and processor saved to {save_directory}")

else:
    print("Skipping training as train_data or val_data is empty.")


# --- Test Set Evaluation ---
if len(test_data) > 0 and 'trainer' in locals(): # Check if test data exists and trainer was initialized
    print("\nEvaluating on Test Set...")
    outputs = trainer.predict(test_data)

    print("\nTest Metrics:")
    print(outputs.metrics)

    # Confusion Matrix
    cm_labels = ["No tumor", "tumor"]
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels, annot_kws={"size": 10}) # Adjusted font size
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', alpha=0.3, cbar=False, xticklabels=cm_labels, yticklabels=cm_labels, annot_kws={"size": 8, "color": "black"}) # Overlay counts

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()
    plt.savefig('test_confusion_matrix.png')
    print("Saved test_confusion_matrix.png")


    # Plot some test images with predictions
    plt.figure() # Create new figure
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 20))
    axes = axes.flatten()
    num_test_images_to_show = min(len(test_images), 18)

    for index in range(num_test_images_to_show):
        # Convert image tensor to numpy array and denormalize
        image_display = test_images[index].transpose(1, 2, 0) # C, H, W -> H, W, C
        mean = np.array(image_mean)
        std = np.array(image_std)
        image_display = std * image_display + mean
        image_display = np.clip(image_display, 0, 1)

        actual_class = id2label[int(test_labels[index])] # Ensure label is int key
        predicted_class = id2label[y_pred[index]]

        ax = axes[index]
        ax.imshow(image_display)
        ax.set_title(f'Actual: {actual_class}\nPredict: {predicted_class}', fontsize=10)
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[num_test_images_to_show:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('test_predictions_sample.png')
    print("Saved test_predictions_sample.png")

else:
    print("Skipping test set evaluation as test_data is empty or trainer was not initialized.")




print("\n--- Testing with external images---")

INTERNET_TEST_BASE_DIR = "C:/Users/sndp_/Downloads/"

# This line constructs the final path: C:/Users/sndp_/Downloads/ + tumor_images
internet_test_dir = os.path.join(INTERNET_TEST_BASE_DIR, 'tumor_images')

if os.path.exists(internet_test_dir) and 'trainer' in locals():
    print(f"Loading images from: {internet_test_dir}")
    internet_test_datagen = ImageDataGenerator(rescale=1./255)

    # Load and label the validation dataset
    try:
        internet_test_generator = internet_test_datagen.flow_from_directory(
            internet_test_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )

        # Process all images from the internet test generator
        print("Applying transforms to external images...")
        internet_test_images, internet_test_labels = apply_transforms(internet_test_generator, val_transforms, process_all=True)

        if len(internet_test_images) > 0:
            internet_test_data = [{"pixel_values": torch.tensor(image), "label": label} for image, label in zip(internet_test_images, internet_test_labels)]

            print("Predicting on external images...")
            internet_outputs = trainer.predict(internet_test_data)

            # Extract predictions
            internet_predictions = internet_outputs.predictions
            print("Raw Predictions (Logits):\n", internet_predictions)

            # Get the predicted class for each input
            internet_predicted_classes = internet_predictions.argmax(axis=1)

            # Map the predicted class indices to their corresponding labels
            internet_predicted_labels = [id2label[class_idx] for class_idx in internet_predicted_classes]

            # Get actual labels from the generator (mapped)
            internet_actual_labels_mapped = [id2label[int(label)] for label in internet_test_labels]

            # Display the predictions
            print("\nPredictions vs Actual (External Images):")
            for i in range(len(internet_test_images)):
                print(f" Image {i+1}: Actual='{internet_actual_labels_mapped[i]}', Predicted='{internet_predicted_labels[i]}'")


            # Display the images and the predicted labels together
            num_internet_images_to_show = min(len(internet_test_images), 10) # Show up to 10
            if num_internet_images_to_show > 0:
                 plt.figure() # Create new figure
                 fig, axes = plt.subplots(1, num_internet_images_to_show, figsize=(num_internet_images_to_show * 3, 4))
                 if num_internet_images_to_show == 1: # Handle single image case
                     axes = [axes]
                 for i in range(num_internet_images_to_show):
                    ax = axes[i]
                    # Convert image tensor to numpy array and denormalize
                    image_display = internet_test_images[i].transpose(1, 2, 0) # C, H, W -> H, W, C
                    mean = np.array(image_mean)
                    std = np.array(image_std)
                    image_display = std * image_display + mean
                    image_display = np.clip(image_display, 0, 1)

                    ax.imshow(image_display)
                    actual_class = internet_actual_labels_mapped[i]
                    ax.set_title(f"Actual: {actual_class}\n Predicted: {internet_predicted_labels[i]}", fontsize=8)
                    ax.axis('off')
                 plt.tight_layout()
                 plt.show()
                 plt.savefig('internet_test_predictions.png')
                 print("Saved internet_test_predictions.png")

        else:
            print("No images processed from the external directory.")

    except FileNotFoundError:
        print(f"Directory not found: {internet_test_dir}. Check the path and ensure 'tumor_images' exists inside {INTERNET_TEST_BASE_DIR}. Skipping external image test.")
    except Exception as e:
        print(f"An error occurred during external image testing: {e}")

else:
    if 'trainer' not in locals():
         print("Skipping external image test because the trainer was not initialized (likely due to empty train/val data).")
    else:
        print(f"Skipping external image test. Directory not found or not set up: {internet_test_dir}")

print("\nScript finished.")