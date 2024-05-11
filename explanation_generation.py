import pandas as pd
import itertools
from multiprocessing import Pool
import numpy as np
import torch

import warnings
warnings.filterwarnings('ignore')

def evaluate_combination(args):
    combo, model, recommendation = args
    # Use numpy directly for manipulation
    temp = recommendation.copy()

    for feat in combo:
        temp[0, feat] = 0  # Update the specific feature in the single row to 0

    prediction = model.predict(temp)[0]
    if prediction == 0:  
        return combo
    return None

def find_features_to_change(model, recommendation):
    # Convert DataFrame to numpy array for faster operation
    recommendation = pd.DataFrame(recommendation).T.to_numpy()
    features = np.where(recommendation > 0)[1]  

    # Generate combinations
    combos = (combo for r in range(1, len(features) + 1) for combo in itertools.combinations(features, r))
    packed_args = [(combo, model, recommendation) for combo in combos]

    with Pool() as pool:
        results = pool.map(evaluate_combination, packed_args)

    for result in results:
        if result is not None:
            return result
    return features


def compute_gradients(model, data_loader):
    model.eval()
    gradients = []
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model.predict(inputs)
        # Ensure outputs and labels have the same dimension
        loss = model.classification_loss(outputs.view(-1, 1), labels.float().view(-1, 1))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Collect gradients of inputs
        input_gradients = inputs.grad.detach().clone() if inputs.grad is not None else None
        gradients.append(input_gradients)
    return gradients


def find_top_influential_instances(model, x_train, test_input, top_k=5):
    model.eval()
    
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=1, shuffle=False)  # Ensure consistent ordering
    train_gradients = compute_gradients(model, train_loader)
    
    # Prepare test input
    if isinstance(test_input, (pd.Series, pd.DataFrame)):
        test_input = torch.tensor(test_input.values, dtype=torch.float32).to(model.device)
    else:
        test_input = test_input.to(model.device)
    test_input.requires_grad = True

    # Forward pass for test input
    test_output = model.predict(test_input).view(-1, 1)
    pseudo_labels = (test_output >= 0.5).float()  # Create pseudo-labels for the test input
    test_loss = model.classification_loss(test_output, pseudo_labels.to(model.device))
    model.zero_grad()
    test_loss.backward()
    test_gradient = test_input.grad.detach() if test_input.grad is not None else None

    # Calculate influence scores using dot product
    influence_scores = []
    for grad in train_gradients:
        if grad is not None:
            influence_score = torch.dot(grad.flatten(), test_gradient.flatten())
            influence_scores.append(influence_score.item())

    # Sort by absolute influence to find the most influential instances
    top_indices = sorted(range(len(influence_scores)), key=lambda i: abs(influence_scores[i]), reverse=True)[:top_k]
    return [(index, influence_scores[index]) for index in top_indices]

