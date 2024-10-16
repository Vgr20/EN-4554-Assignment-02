#import necessary modules
import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets,transforms 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from tqdm import tqdm


#Define the directory to store the dataset
data_dir = './caltech101'

#Define the transformations for the dataset.
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
#Download the dataset
dataset = datasets.Caltech101(data_dir,download=True,transform=transform)
labels = dataset.y

######################--------Dataset Analysis--------######################

def print_statistics(dataset):
    num_samples = len(dataset)
    num_classes = len(dataset.annotation_categories)
    class_names = dataset.annotation_categories
    samples_per_class = dataset.y

    samples_per_class = np.bincount(samples_per_class)
    print(f'Number of samples in the dataset: {num_samples}')
    print(f'Number of classes in the dataset: {num_classes}')
    print(f'Class names: {class_names}')

    for class_name, samples in zip(class_names, samples_per_class):
        print(f' {class_name}, {samples}')

    def plot_histogram(samples_per_class, class_names):
        #Plotting the histogram
        plt.figure(figsize=(20,8))
        plt.bar(class_names,samples_per_class)
        plt.xticks(rotation=90, ha='right') # Rotate the x-axis labels to make them readable and avoid overlapping
        #Adding frequency on top of the bars
        for i, samples in enumerate(samples_per_class):
            plt.text(i, samples, str(samples), ha='center', va='bottom')
        plt.xlabel('Class Names')
        plt.ylabel('Number of samples')
        plt.title('Number of samples per class')

        plt.tight_layout()
        plt.show()

    plot_histogram(samples_per_class, class_names)

# print_statistics(dataset)

######################--------Train_Test Split Analysis--------######################
print("------------------------Train_Test Split Started------------------------")

# Train test split using Stratified split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.3333, stratify=labels, random_state=42)

print("------------------------Train_Test Split Completed------------------------")

print("Train dataset length:", len(train_dataset))
print("Test dataset length:", len(test_dataset))

# Create the dataloaders
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

# Visualize the images
def data_visualization(train_loader):
    # Get a batch of training data
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Visualize the images
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize the image
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    # Show images in the first batch
    imshow(torchvision.utils.make_grid(images))
    print("Labels:", labels)

# data_visualization(train_loader)
# data_visualization(test_loader)

######################--------Resnet Models--------######################

#Load the pre-trained model
model = resnet152(weights='IMAGENET1K_V1')
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is loaded on {device}")

######################--------Extracting Embeddings--------######################

print("------------------------Extracting Embeddings Started------------------------")

#Get embeddings for the images
def get_embeddings(loader,model):
    embeddings = []
    labels = []
    for images, target in tqdm(loader):
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
        embeddings.append(output.cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

train_embeddings, train_labels = get_embeddings(train_loader, model)
test_embeddings, test_labels = get_embeddings(test_loader, model)

train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], -1)

print("------------------------Extracting Embeddings Completed------------------------")

######################--------KNN Classifier--------######################

print("------------------------KNN Classifier Started------------------------")

# Define and implement the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)
accuracy = knn.score(test_embeddings, test_labels) * 100
print(f'Accuracy of the KNN classifier: {accuracy} %')

print("------------------------KNN Classifier Completed------------------------")
print(f"Training and Testing Completed Successfully with an accuracy score of {accuracy} %")


from sklearn.linear_model import LogisticRegression

print("------------------------Logistic Regression Started------------------------")

# Define the logistic regression classifier
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Fit the logistic regression model on the training embeddings and labels
log_reg.fit(train_embeddings, train_labels)

# Evaluate the model on the test embeddings
test_accuracy = log_reg.score(test_embeddings, test_labels) * 100

print(f'Accuracy of the Logistic Regression classifier: {test_accuracy} %')

print("------------------------Logistic Regression Completed------------------------")
print(f"Training and Testing Completed Successfully with an accuracy score of {test_accuracy} %")
