import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

class AutoEncoderClassifier():

    def __init__(self,embedder):
        self.embedder = embedder
        
    def explain(self, X: np.array, y: np.array, raw_data: pd.DataFrame) -> pd.Series:
        dummy_data = np.zeros(len(raw_data.columns))  
        return pd.Series(dummy_data, index=raw_data.columns)
        
    def fit(self, X, y, num_epochs = 5, *args, **kwargs):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(self.embedder.parameters(), lr=0.001, weight_decay=0.01)
        return self.embedder.fit(num_epochs, dataloader, optimizer)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        probabilities = self.embedder.predict(X_tensor).detach()
        labels = (probabilities >= 0.5).float()
        return labels.numpy()

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.embedder.predict_proba(X_tensor)
    
class EnhancedLR(nn.Module):
    
    def __init__(self, input_dim):
        super(EnhancedLR , self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)

    def predict_proba(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return torch.cat((1 - output, output), dim=1).numpy()

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, reconstruction_loss, classification_loss, dropout_rate = 0.15):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.reconstruction_loss = reconstruction_loss
        self.classification_loss = classification_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, embedding_dim),
            )

        self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, input_dim),
                nn.Sigmoid(),
            )
    
        self.classifier = EnhancedLR(embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def predict(self, x):
        x = self.encoder(x)
        prediction = self.classifier(x)
        return prediction
    
    def predict_proba(self, x):
        x = self.encoder(x)
        predict_proba = self.classifier.predict_proba(x)
        return predict_proba
    
    def unsupervised_train(self, num_epochs, train_dataloader, val_dataloader, optimizer):

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            
            # Training phase
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch[0].to(self.device)
                outputs = self(inputs)
                loss = self.reconstruction_loss(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs = batch[0].to(self.device)
                    outputs = self(inputs)
                    val_loss = self.reconstruction_loss(outputs, inputs)
                    total_val_loss += val_loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader)
            
            print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    def fit(self, num_epochs, train_dataloader, optimizer):
        
        # L1 Regularization strength
        lambda_l1 = 0.2

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            total_clasification_loss = 0
            total_recon_loss = 0
            
            # Training phase
            for inputs, labels in train_dataloader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)

                # Fine-tune the embeddings
                reconstruction_loss = self.reconstruction_loss(outputs, inputs)

                # Train the classifier 
                predictions = self.predict(inputs)
                classification_loss = self.classification_loss(predictions, labels.squeeze().view(-1,1))

                # L1 regularization
                l1_reg = torch.tensor(0.)
                for param in self.classifier.parameters():
                    l1_reg += torch.norm(param, 1)

                # Optimize Both 
                loss = reconstruction_loss + classification_loss +  ( l1_reg * lambda_l1 ) 
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_clasification_loss = classification_loss.item()
                total_recon_loss = reconstruction_loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_class_loss = total_clasification_loss / len(train_dataloader)
            avg_recon_loss = total_recon_loss / len(train_dataloader)
            
            print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Recon Loss: {avg_recon_loss}, Classification Loss: {avg_class_loss}')