import shutil
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.formula.api as smf
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# src_pth = r"E:\INTERNSHIPS\Feynn Labs Internships\Project 1\mcdonalds.csv"
# dest_pth = os.path.join(os.getcwd(), 'mcdonalds.csv')
# shutil.copy(src_pth, dest_pth)
mcdonalds = pd.read_csv("mcdonalds.csv")
# print(mcdonalds.columns.tolist())
# print(mcdonalds.shape)
# print(mcdonalds.head(3))

MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == 'Yes').astype(int))
# Calculate column means and round to 2 decimal places
# column_means = MD_x.mean().round(2)
# print(column_means)

#MD_pca = PCA().fit(MD_x)
# Print summary
#print("Importance of components:")
# print("PC\tStandard deviation\tProportion of Variance\tCumulative Proportion")
# for i in range(len(MD_pca.explained_variance_ratio_)):
#     print(f"PC{i+1}\t{MD_pca.explained_variance_[i]:.4f}\t\t\t{MD_pca.explained_variance_ratio_[i]:.4f}\t\t\t{np.cumsum(MD_pca.explained_variance_ratio_)[i]:.4f}")
# Print standard deviations
# print("Standard deviations (1, .., p=11):")
# for sd in MD_pca.explained_variance_:
#     print(f"{sd:.1f}", end=" ")
# Plot principal components
# plt.scatter(MD_pca.transform(MD_x)[:, 0], MD_pca.transform(MD_x)[:, 1], c='grey')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('Plotting of first 2 Principal Components')
# plt.show()

np.random.seed(1234)
# cluster_results = {}
# for k in range(2,9):
#      kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
#      kmeans.fit(MD_x)
#      kmeans_labels = kmeans.labels_
#      cluster_results[k]=kmeans.labels_
# MD_km28 = cluster_results
# print(MD_km28.keys())
#
# sum_of_squares = []
# for k, labels in MD_km28.items():
#     kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
#     kmeans.fit(MD_x)
#     sum_of_squares.append(kmeans.inertia_)
#
# # Plot the WCSS versus number of segments
# plt.plot(range(2, 9), sum_of_squares, marker='o')
# plt.xlabel('Total number of Segments')
# plt.ylabel('Within-Cluster Sum of Squares')
# plt.title('Elbow Method for Optimal Number of Segments')
# plt.grid(True)
# plt.show()

# nboot = 100  # Number of bootstrap samples
# nreps = 10  # Number of repetitions
# n_clusters_range = range(2, 9)
#
# boot_results = {}
# for k in n_clusters_range:
#     boot_labels = []
#     true_labels = mcdonalds['VisitFrequency']
#     for _ in range(nboot):
#         # Sample with replacement
#         indices = np.random.choice(len(MD_x), len(MD_x), replace=True)
#         boot_data = MD_x[indices]
#
#         # Perform K-means clustering
#         kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
#         kmeans.fit(boot_data)
#         boot_labels.append(kmeans.labels_)
#
#     boot_results[k] = boot_labels
#
# # Calculate adjusted Rand index for each number of segments
# adjusted_rand_indices = []
# for k, boot_labels_list in boot_results.items():
#     ari_values = []
#     for boot_labels in boot_labels_list:
#         ari = adjusted_rand_score(true_labels, boot_labels)  # Assuming true labels are available
#         ari_values.append(ari)
#     adjusted_rand_indices.append(np.mean(ari_values))
#
# # Plot adjusted Rand index versus number of segments
# plt.plot(n_clusters_range, adjusted_rand_indices, marker='o')
# plt.xlabel('Number of Segments')
# plt.ylabel('Adjusted Rand Index')
# plt.title('Bootstrap Clustering: Adjusted Rand Index vs. Number of Segments')
# plt.grid(True)
# plt.show()

# cluster_assignments = MD_km28.cluster
# # Extracting the data associated with cluster "4"
# cluster_4_indices = np.where(cluster_assignments == 4)[0]
# cluster_4_data = MD_x.iloc[cluster_4_indices]

# Plotting the histogram
# plt.figure(figsize=(8, 6))
# plt.hist(cluster_4_data['4'], bins=20, range=(0, 1))
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of Cluster 4')
# plt.grid(True)
# plt.show()

# results = []
# for k in range(2,9):
#     gmm = GaussianMixture(n_components=k,n_init=10, verbose=0, covariance_type='full', random_state=1234)
#
#     # Fit the model to the data
#     gmm.fit(MD_x)
#     gmm_labels = gmm.predict(MD_x)
#     # Store the results
#     results.append({
#         'k': k,
#         'log_likelihood': gmm.lower_bound_,
#         'bic': gmm.bic(MD_x),
#         'aic': gmm.aic(MD_x),
#     })

# Display the results
# for result in results:
#     print(f"k={result['k']}: Log Likelihood={result['log_likelihood']}, BIC={result['bic']}, AIC={result['aic']}")

# kmeans_clusters = Counter(kmeans_labels)
# mixture_clusters = Counter(gmm_labels)
#
# # Print the table
# print("mixture")
# print("kmeans", end=" ")
# for k, v in mixture_clusters.items():
#     print(k, end=" ")
# print()
# for k1, v1 in kmeans_clusters.items():
#     print(k1, end=" ")
#     for k2, v2 in mixture_clusters.items():
#         print(v2 if k1 == k2 else 0, end=" ")
#     print()
# cluster_combinations = Counter(zip(kmeans_labels, gmm_labels))
#
# # Print the table
# print("mixture")
# print("kmeans", end=" ")
# for k, v in cluster_combinations.items():
#     print(k[0], end=" ")
# print()
# for k1, v1 in Counter(kmeans_labels).items():
#     print(k1, end=" ")
#     for k2, v2 in Counter(gmm_labels).items():
#         print(cluster_combinations[(k1, k2)], end=" ")
#     print()
# MD_m4a_log_likelihood = gmm_labels.score(MD_x)
# print("log Lik. (gmm_labels):", MD_m4a_log_likelihood)
# mcdonalds_like_counts = mcdonalds['Like'].value_counts().sort_index(ascending=False)
# print("Original counts:")
# print(mcdonalds_like_counts)
#
# # Create a new column 'Like.n' by subtracting each value from 6
# mcdonalds['Like.n'] = 6 - mcdonalds['Like'].astype(int)
#
# # Count the values in the new column
# mcdonalds_like_n_counts = mcdonalds['Like.n'].value_counts().sort_index(ascending=False)
# print("Modified counts:")
# # print(mcdonalds_like_n_counts)
# column_names = mcdonalds.columns[0:11]
# f = "+".join(column_names)
# f = "Like.n ~ "+f
# #print(f)
# mcdonalds_numeric = mcdonalds.select_dtypes(include=['int', 'float'])
# gmm = GaussianMixture(n_components=2, n_init=10)
# MD_reg2 = gmm.fit(mcdonalds_numeric)
#
# # Print cluster sizes
# cluster_labels = gmm.predict(mcdonalds_numeric)
#
# # Count cluster sizes
# cluster_sizes = Counter(cluster_labels)
# MD_ref2 = LogisticRegression()
# MD_ref2.fit(mcdonalds_numeric, MD_reg2.predict(mcdonalds_numeric))

# Summary of the refitted model
# print("Comp.1:")
# print("Estimate\tStd. Error\tz value\t\tPr(>|z|)")
# for feature, coef, std_err in zip(mcdonalds_numeric.columns, MD_ref2.coef_[0], np.std(mcdonalds_numeric, 0)):
#     z_value = coef / std_err
#     p_value = 2 * (1 - norm.cdf(abs(z_value)))
#     print(f"{feature}\t{coef:.6f}\t{std_err:.6f}\t{z_value:.6f}\t{p_value:.6f}")
# print("Comp.2:")
# print("Estimate\tStd. Error\tz value\t\tPr(>|z|)")
# for feature, coef, std_err in zip(mcdonalds_numeric.columns, MD_ref2.coef_[1], np.std(mcdonalds_numeric, 0)):
#     z_value = coef / std_err
#     p_value = 2 * (1 - norm.cdf(abs(z_value)))
#     print(f"{feature}\t{coef:.6f}\t{std_err:.6f}\t{z_value:.6f}\t{p_value:.6f}")

# MD_ref2 = LogisticRegression()
# MD_ref2.fit(mcdonalds_numeric, cluster_labels)
# logit_model = smf.logit('cluster_labels ~ ' + '+'.join(mcdonalds_numeric.columns), data=mcdonalds_numeric)
# logit_results = logit_model.fit()
#
# # Plot partial regression plots
# fig = smg.regressionplots.plot_partregress_grid(logit_results)
# plt.show()

# Print cluster sizes
# print("Cluster sizes:")
# for label, size in cluster_sizes.items():
#     print(f"Cluster {label + 1}: {size}")

# distances = pdist(MD_x.T)
# # Perform hierarchical clustering
# MD_vclust = hierarchy.linkage(distances, method='single')
like_counts = mcdonalds['Like'].value_counts()
# plt.figure(figsize=(10, 6))
# sns.barplot(x=like_counts.index, y=like_counts.values, palette='viridis')
# plt.title('Distribution of Like')
# plt.xlabel('Like')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.grid(True)
# plt.show()
visit_frequency = mcdonalds['VisitFrequency'].astype(float)
visit_mean_by_cluster = visit_frequency.groupby(like_counts).mean()

print(visit_mean_by_cluster)







# MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=1234)
# MD_reg2.fit(mcdonalds)
# print(MD_reg2.predict(mcdonalds))
# print(MD_reg2.weights_)




























