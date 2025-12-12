import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_kmeans_clustering(file_path):
    """
    Performs K-means clustering on a mall customer dataset.

    Args:
        file_path (str): The path to the 'Mall_Customers.csv' file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the file is in the same directory as this script.")
        return

    # Display initial information about the dataset
    print("Initial Data Information:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()

    # Select features for clustering
    # We will use 'Annual Income (k$)' and 'Spending Score (1-100)'
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the data for K-means
    # Scaling is crucial as K-means is a distance-based algorithm
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Create and display the Elbow Method plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method to Determine Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.show() 
    
    # Based on the typical elbow plot for this dataset, K=5 is the optimal number.
    optimal_k = 5
    print(f"\nDetermining optimal number of clusters to be {optimal_k} based on the elbow plot.")
    
    # Apply K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create and display the scatter plot of the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
    # Plotting the centroids on the original scale for better interpretability
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('Customer Segments by K-means Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show() 

    # Display the resulting clusters and counts
    print("\nDataFrame with Cluster Labels:")
    print(df.head())
    print("\nNumber of customers in each cluster:")
    print(df['Cluster'].value_counts().sort_index())

    # Save the updated DataFrame to a new CSV file
    df.to_csv('clustered_customers.csv', index=False)
    print("\nUpdated DataFrame saved to 'clustered_customers.csv'")

# Run the function with the correct file path
if __name__ == "__main__":
    file_name = 'Mall_Customers.csv'
    perform_kmeans_clustering(file_name)