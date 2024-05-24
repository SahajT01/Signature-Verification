import matplotlib.pyplot as plt
import pandas as pd

# Data for the models
models = ['SVM', 'HMM', 'Neural Network', 'Siamese NN']
accuracy = [78, 75, 81, 96.2]

# Creating the DataFrame
df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy
})

# Plotting Accuracy
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy (%)')
ax1.bar(df['Model'], df['Accuracy'], color='tab:blue', alpha=0.6)
ax1.tick_params(axis='y')
fig.tight_layout()
plt.title('Model Accuracy Comparison')
plt.savefig('Model_Accuracy_Comparison.svg', format='svg')
plt.show()
