# command_models.py

import numpy as np
import torch
import torch.nn as nn
import os
import string


class UtteranceClassifier:
    def __init__(self, model_path, input_size, hidden_size, num_classes, client):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.client = client
        
        # Initialize and load the model
        self.model = self._initialize_model(client)
        self._load_model(model_path)
        
    def _initialize_model(self, client):
        class Classifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(Classifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        
        model = Classifier(self.input_size, self.hidden_size, self.num_classes)
        print("> Model Initialized")
        return model
    
    def _load_model(self, model_path):
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("> Model Loaded")
        
    def preprocess_utterance(self, text, model="text-embedding-3-large"):
        text = text.replace("\n", " ")
        text = text.lower()
        text = text.strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def predict(self, utterance):
        embedding = self.preprocess_utterance(utterance)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(embedding_tensor)
            _, predicted = torch.max(output, 1)
        
        return predicted.item()

# # Example usage
# if __name__ == "__main__":
#     model_path = 'saved_modelsV8/model_epoch_4.pt'  # Path to the trained model
#     input_size = 128  # Example input size, replace with your actual input size
#     hidden_size = 64
#     num_classes = 3

#     classifier = UtteranceClassifier(model_path, input_size, hidden_size, num_classes)
#     utterance = "This is an example utterance"
#     predicted_class = classifier.predict(utterance)
#     print(f"Predicted class for the utterance: {predicted_class}")
