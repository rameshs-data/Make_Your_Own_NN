# NN From Scratch - Notes & Learnings

## Epoch 2: Rebuilding Neural Network from Scratch

### 1. **Neural Network Structure**
   - **Layer Structure**: A neural network typically consists of input layers, hidden layers, and output layers.
   - **Weights & Biases**: Initializing weights randomly and bias values to zero or small constants.
   - **Forward Pass**: The process where inputs are passed through the network to generate predictions.

### 2. **Activation Functions**
   - **ReLU**: Rectified Linear Unit; preferred for deep networks due to less likelihood of vanishing gradients.
   - **Sigmoid**: Used in binary classification problems for its probabilistic output.
   - **Tanh**: Often used in hidden layers, outputs between -1 and 1.

### 3. **Backpropagation**
   - **Goal**: Update weights to minimize the error between predictions and true labels.
   - **Process**:
     - Calculate the error (loss) at the output layer.
     - Backtrack through layers using chain rule to compute gradients.
     - Update weights using gradient descent to minimize loss.
   - **Gradient Descent**: Iterative optimization process, aiming to find the minimum of the loss function.

### 4. **Optimization**
   - **Stochastic Gradient Descent (SGD)**: Updates parameters after every data point.
   - **Batch Gradient Descent**: Uses a batch of training data for parameter updates.

### 5. **Challenges**
   - **Overfitting**: Early model trials resulted in high training accuracy but poor generalization to new data.
     - **Solution**: Applied regularization methods like dropout.
   - **Vanishing Gradients**: Found that deeper networks with Sigmoid activation led to gradient issues.
     - **Solution**: Switched to ReLU activation to mitigate this problem.

### 6. **Performance Evaluation**
   - **Loss Function**: Used Mean Squared Error (MSE) for regression tasks and Cross-Entropy for classification tasks.
   - **Accuracy**: Regularly evaluated accuracy after each epoch to track model improvements.

### 7. **Debugging**
   - **Issue**: Gradients were vanishing after the first few layers.
     - **Solution**: Switched to ReLU and initialized weights using Xavier initialization.

### 8. **Next Steps**
   - Implement Adam optimizer and compare it with standard gradient descent.
   - Explore building a neural network with more hidden layers and dropout layers for regularization.

---
## Additional Resources
- [Backpropagation Explained](link)
- [Understanding Neural Networks](link)
- [Introduction to Optimization Techniques](link)
