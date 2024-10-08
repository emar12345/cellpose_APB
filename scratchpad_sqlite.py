import sqlite3
import matplotlib.pyplot as plt
import numpy as np

"""
reads the database and outputs some nice graphs. these are the sample queries [1,2,3]
"""

# # query 1: writes [Model Name:t_whatever, Epoch X - AP @ threshold [a,b,c]] to console
# # Connect to the SQLite database
# conn = sqlite3.connect('identifier.sqlite')
# cursor = conn.cursor()
#
# # Define the list of epoch values
# epoch_values = list(range(1, 102, 10)) + list(range(201, 902, 100)) + [999]
# model_name = "cellpose_residual_on_style_on_concatenation_off_t_0_2023_08_08_19_46_32.613015"
# print(f"MODEL NAME: {model_name}")
#
# for epoch in epoch_values:
#     # Define your query
#     query = f"""
#     SELECT AVG(AP_threshold_0_50), AVG(AP_threshold_0_75), AVG(AP_threshold_0_90)
#     FROM test_results
#     WHERE model_name = "{model_name}" AND epoch = "{epoch}" AND test_name NOT LIKE "%empty%";
#     """
#
#     # Execute the query
#     cursor.execute(query)
#
#     # Fetch the results
#
#     average_scores = cursor.fetchone()
#     # Print the average scores for the current epoch
#     print(f"Epoch {epoch} - Average AP_threshold_0_50:", average_scores[0])
#     print(f"Epoch {epoch} - Average AP_threshold_0_75:", average_scores[1])
#     print(f"Epoch {epoch} - Average AP_threshold_0_90:", average_scores[2])
#     print()
#
# # Close the connection
# conn.close()

# # query 2 : plot models' empty vacuole score over epoch
#
#
# conn = sqlite3.connect('identifier.sqlite')
# cursor = conn.cursor()
#
# # Define the epoch values and model name
# epoch_values = list(range(1, 102, 10)) + list(range(201, 902, 100)) + [999]
# t_0 = "cellpose_residual_on_style_on_concatenation_off_t_0_2023_08_08_19_46_32.613015"
# t_1 = 'cellpose_residual_on_style_on_concatenation_off_t_1_2023_08_08_20_10_50.928119'
# t_2 = 'cellpose_residual_on_style_on_concatenation_off_t_2_2023_08_08_20_57_59.141163'
# t_3 = 'cellpose_residual_on_style_on_concatenation_off_t_3_2023_08_09_03_22_56.754794'
#
# # Prepare data for plotting
# model_names = [t_0, t_1, t_2, t_3]
# model_name_names = {
#     t_0: "Model 0",
#     t_1: "Model 1",
#     t_2: "Model 2",
#     t_3: "Model 3"
# }
#
# # Dictionary to store bool averages for each model
# model_bool_averages = {}
#
# for model_name in model_names:
#     bool_averages = []
#     for epoch in epoch_values:
#         # Define your query
#         query = f"""
#         SELECT bool_values
#         FROM empty_results
#         WHERE model_name = "{model_name}" AND epoch = {epoch};
#         """
#
#         # Execute the query
#         cursor.execute(query)
#
#         # Fetch the results
#         result = cursor.fetchall()
#
#         # Check if result is not None
#         bool_values = [value == 'True' for value, in result]
#         average = sum(bool_values) / len(bool_values)
#         bool_averages.append(average)
#
#     model_bool_averages[model_name] = bool_averages
#
# # Close the connection
# conn.close()
#
# # Create a single plot for all models
# plt.figure()
#
# for model_name in model_names:
#     plt.plot(epoch_values, model_bool_averages[model_name], marker='o', label=model_name_names[model_name])
#
# plt.title("Average Empty Vacuole Score Over Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Average Empty Vacuole Score")
# plt.grid()
#
# # Set custom grid intervals for y-axis
# yticks_interval = 0.05
# yticks = np.arange(0, 1.1, yticks_interval)
# plt.yticks(yticks)
# plt.ylim(0, 1.00)  # Set maximum y-axis value to 1.00
#
# # Set custom grid intervals for x-axis
# xticks_interval = 100
# xticks = np.arange(0, max(epoch_values) + xticks_interval, xticks_interval)
# plt.xticks(xticks)
#
# plt.legend()
# plt.show()


# #query 3.  3 plots for threshold 0.50, threshold 0.75, and threshold 0.90 with model AP score over epoch.
#
# import sqlite3
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Connect to the SQLite database
# conn = sqlite3.connect('identifier.sqlite')
# cursor = conn.cursor()
#
# # Define the epoch values and model names
# epoch_values = list(range(1, 102, 10)) + list(range(201, 902, 100)) + [999]
# t_0 = "cellpose_residual_on_style_on_concatenation_off_t_0_2023_08_08_19_46_32.613015"
# t_1 = 'cellpose_residual_on_style_on_concatenation_off_t_1_2023_08_08_20_10_50.928119'
# t_2 = 'cellpose_residual_on_style_on_concatenation_off_t_2_2023_08_08_20_57_59.141163'
# t_3 = 'cellpose_residual_on_style_on_concatenation_off_t_3_2023_08_09_03_22_56.754794'
#
# # Corresponding names for model names
# model_name_names = {
#     t_0: "Model 0",
#     t_1: "Model 1",
#     t_2: "Model 2",
#     t_3: "Model 3"
# }
#
# # Thresholds for plotting
# # Thresholds for plotting
# thresholds = ['0_50', '0_75', '0_90']
# threshold_mapping = {
#     '0_50': '0.50',
#     '0_75': '0.75',
#     '0_90': '0.9'
# }
#
# # Prepare data for plotting
# data = {model_name: {threshold: [] for threshold in thresholds} for model_name in model_name_names.keys()}
#
# for model_name in model_name_names.keys():
#     for epoch in epoch_values:
#         for threshold in thresholds:
#             # Define your query
#             query = f"""
#             SELECT AVG(AP_threshold_{threshold})
#             FROM test_results
#             WHERE model_name = "{model_name}" AND epoch = "{epoch}" AND test_name NOT LIKE "%empty%";
#             """
#
#             # Execute the query
#             cursor.execute(query)
#
#             # Fetch the results
#             average_score = cursor.fetchone()[0]
#             data[model_name][threshold].append(average_score)
#
# # Close the connection
# conn.close()
#
# # Create separate plots for each threshold
# for threshold in thresholds:
#     plt.figure()
#
#     for model_name in model_name_names.keys():
#         plt.plot(epoch_values, data[model_name][threshold], marker='o', label=f"{model_name_names[model_name]}")
#
#     plt.title(f"Average AP Scores Over Epoch - IoU Threshold {threshold_mapping[threshold]}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Average AP Score")
#     plt.grid()
#
#     # Set custom grid intervals for y-axis
#     yticks_interval = 0.05
#     yticks = np.arange(0, 1.1, yticks_interval)
#     plt.yticks(yticks)
#
#     # Set custom grid intervals for x-axis
#     xticks_interval = 100
#     xticks = np.arange(0, max(epoch_values) + xticks_interval, xticks_interval)
#     plt.xticks(xticks)
#
#     plt.legend()
#     plt.show()
#
