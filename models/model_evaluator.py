import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import os
import json
# Import necessary sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, balanced_accuracy_score # Added balanced_accuracy
)
# Ensure correct relative imports if evaluator is run from a parent directory
# Or adjust paths if run directly from within 'models' directory
try:
    from models.decision_tree import DecisionTreeVacationRecommender
    from models.knn_model import KNNVacationRecommender
    from models.iterative_deepening import IDDFSVacationRecommender
    from models.genetic_algorithm import GeneticVacationRecommender
    from models.a_star_search import AStarVacationRecommender
except ImportError:
    # Handle cases where the script might be run from the models directory itself
    from decision_tree import DecisionTreeVacationRecommender
    from knn_model import KNNVacationRecommender
    from iterative_deepening import IDDFSVacationRecommender
    from genetic_algorithm import GeneticVacationRecommender
    from a_star_search import AStarVacationRecommender


# Configure logger (consider moving to a central logging setup)
# Basic configuration for demonstration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_save_dir='models/saved_models'):
        """
        Initializes the ModelEvaluator.

        Args:
            model_save_dir (str): Directory to save/load trained models.
        """
        self.model_configs = { # Renamed from self.models to avoid confusion with loaded models
            'Decision Tree': DecisionTreeVacationRecommender(),
            'KNN': KNNVacationRecommender(),
            'IDDFS': IDDFSVacationRecommender(),
            'Genetic Algorithm': GeneticVacationRecommender(),
            'A* Search': AStarVacationRecommender()
        }
        self.trained_models = {} # To store trained model instances
        self.results = {}
        self.model_save_dir = model_save_dir
        self.target_column = 'recommended_vacation' # Define target column name

        # Ensure model saving directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)
        logger.info(f"Model save directory set to: {self.model_save_dir}")

    def train_and_evaluate_all_models(self, df, test_size=0.25, random_state=42):
        """
        Splits data, trains all models on the training set,
        and evaluates them on the test set.

        Args:
            df (pd.DataFrame): The complete dataset.
            test_size (float): Proportion of the dataset to use for the test set.
            random_state (int): Random state for reproducibility of the split.
        """
        logger.info("===== Training and Evaluation Process Starting =====")

        if df is None or df.empty:
            logger.error("Input DataFrame is empty or None. Aborting.")
            return

        if self.target_column not in df.columns:
             logger.error(f"Target column '{self.target_column}' not found in DataFrame. Aborting.")
             # Attempt to infer target column if it's the last one (less robust)
             # inferred_target = df.columns[-1]
             # logger.warning(f"Attempting to use inferred target column: {inferred_target}")
             # self.target_column = inferred_target
             return # Stop if target not found


        # --- 1. Data Splitting ---
        logger.info(f"Splitting data into training and testing sets (Test Size: {test_size}, Random State: {random_state})...")
        try:
            # Ensure target column exists before dropping
            features = df.drop(self.target_column, axis=1)
            labels = df[self.target_column]

            # Stratify ensures class distribution is similar in train and test sets
            train_df, test_df = train_test_split(
                df, # Split the original DataFrame to keep features and labels together initially
                test_size=test_size,
                random_state=random_state,
                stratify=labels # Use labels for stratification
            )
            logger.info(f"Training set size: {train_df.shape[0]} samples")
            logger.info(f"Testing set size: {test_df.shape[0]} samples")

        except Exception as e:
            logger.error(f"Error during data splitting: {e}", exc_info=True)
            return

        # --- 2. Training Models ---
        logger.info("--- Starting Model Training Phase ---")
        self.trained_models = {} # Reset trained models dict
        training_times = {}

        # ASSUMPTION: Each model's .train() method handles necessary preprocessing internally
        # (e.g., fitting scalers/encoders only on the training data it receives).
        for model_name, model_instance in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            start_time = time.time()
            try:
                # Train ONLY on the training set
                model_instance.train(train_df.copy()) # Pass a copy to avoid potential modifications
                end_time = time.time()
                training_time = end_time - start_time
                training_times[model_name] = training_time
                self.trained_models[model_name] = model_instance # Store the trained instance
                logger.info(f"{model_name} training completed. Duration: {training_time:.2f} seconds")

                # Save the trained model (using model's internal save method)
                # Ensure the model's save method uses self.model_save_dir or accepts a path
                # Example: model_instance.save_model(directory=self.model_save_dir)
                # If save_model doesn't take args, ensure it saves to the correct place.
                try:
                     # Attempt to call save_model if it exists
                     if hasattr(model_instance, 'save_model') and callable(model_instance.save_model):
                          model_instance.save_model() # Assuming it saves to a predefined path or uses a path set during init
                          # Ideally: model_instance.save_model(os.path.join(self.model_save_dir, f"{model_name.lower().replace(' ', '_')}.pkl"))
                          logger.info(f"{model_name} model saved.")
                     else:
                          logger.warning(f"{model_name} does not have a callable 'save_model' method.")

                except Exception as save_e:
                     logger.error(f"Error saving {model_name} model: {save_e}", exc_info=True)


            except Exception as e:
                logger.error(f"Error during {model_name} training: {e}", exc_info=True)
                self.trained_models[model_name] = None # Mark as failed
                training_times[model_name] = None

        logger.info("--- Model Training Phase Completed ---")

        # --- 3. Evaluating Models ---
        # Pass ONLY the test set and training times to the evaluation method
        self.evaluate_all_models(test_df.copy(), training_times) # Pass a copy

        # --- 4. Comparing and Reporting ---
        logger.info("--- Generating Comparison Report ---")
        comparison_df = self.compare_models() # Uses self.results populated by evaluate_all_models

        # Save comparison metrics to JSON
        if comparison_df is not None and not comparison_df.empty:
             # Create evaluation directory if it doesn't exist
             os.makedirs('evaluation', exist_ok=True)
             metrics_json_path = 'evaluation/metrics.json'
             try:
                  metrics_json = comparison_df.to_dict(orient='records')
                  with open(metrics_json_path, 'w') as f:
                       json.dump(metrics_json, f, indent=4) # Use indent for readability
                  logger.info(f"Evaluation metrics saved to: {metrics_json_path}")
             except Exception as e:
                  logger.error(f"Error saving metrics to JSON: {e}", exc_info=True)

        logger.info("===== Training and Evaluation Process Finished =====")

    def evaluate_all_models(self, test_df, training_times):
        """
        Evaluates all *trained* models on the provided *test* dataset.

        Args:
            test_df (pd.DataFrame): The dataset reserved for testing (unseen during training).
            training_times (dict): Dictionary mapping model names to their training times.
        """
        logger.info("--- Starting Model Evaluation Phase ---")
        logger.info(f"Evaluating on test set with {test_df.shape[0]} samples.")

        if test_df is None or test_df.empty:
             logger.error("Test DataFrame is empty or None. Cannot evaluate.")
             return

        if self.target_column not in test_df.columns:
             logger.error(f"Target column '{self.target_column}' not found in test DataFrame. Cannot evaluate.")
             return

        # Create evaluation directories if they don't exist
        os.makedirs('evaluation', exist_ok=True)
        os.makedirs('evaluation/confusion_matrices', exist_ok=True) # Subdirectory for plots

        self.results = {} # Reset results for this evaluation run

        # Get actual labels from the test set
        ground_truth_labels = test_df[self.target_column].tolist()
        unique_labels = sorted(test_df[self.target_column].unique()) # For confusion matrix labels

        # Check class distribution in the test set
        destination_counts = test_df[self.target_column].value_counts()
        logger.info(f"Test set destination distribution:\n{destination_counts.to_string()}")

        # ASSUMPTION: Each model's .predict() method handles necessary preprocessing internally
        # using parameters learned during training (e.g., loading saved scalers/encoders).
        for model_name, model in self.trained_models.items():
            if model is None:
                logger.warning(f"Skipping evaluation for {model_name} as it was not trained successfully.")
                # Store placeholder results
                self.results[model_name] = {
                    'training_time': training_times.get(model_name),
                    'model': None,
                    'accuracy': 'Train Error',
                    # ... other metrics as 'Train Error'
                }
                continue

            logger.info(f"Evaluating {model_name}...")
            start_time = time.time()
            predictions = []
            confidences = []
            valid_predictions = 0

            # Iterate through test set rows to get predictions
            # Note: Batch prediction would be faster if models support it.
            for _, row in test_df.iterrows():
                 # Prepare input for model's predict method (assuming dict format)
                 # Adjust keys based on actual features used by models
                 user_preferences = row.drop(self.target_column).to_dict()
                 # Convert types if necessary (e.g., budget, duration to float) - Do safely
                 for key in ['budget', 'duration']:
                      if key in user_preferences:
                           try:
                                user_preferences[key] = float(user_preferences[key])
                           except (ValueError, TypeError):
                                logger.warning(f"Could not convert {key} '{user_preferences[key]}' to float for {model_name}. Using NaN or original value.")
                                user_preferences[key] = np.nan # Or handle as needed by model

                 try:
                      # Assuming predict returns a list of dicts or a single dict
                      prediction_result = model.predict(user_preferences) # Removed top_n, adjust if needed

                      # Extract top prediction and confidence
                      top_prediction_dest = None
                      top_confidence = 0.0

                      if isinstance(prediction_result, list) and prediction_result:
                           top_pred_obj = prediction_result[0]
                           if isinstance(top_pred_obj, dict) and 'destination' in top_pred_obj:
                                top_prediction_dest = top_pred_obj['destination']
                                top_confidence = top_pred_obj.get('confidence', 0.0)
                      elif isinstance(prediction_result, dict) and 'destination' in prediction_result:
                           top_prediction_dest = prediction_result['destination']
                           top_confidence = prediction_result.get('confidence', 0.0)
                      elif isinstance(prediction_result, str): # Handle models returning only destination string
                           top_prediction_dest = prediction_result
                           top_confidence = 0.0 # Confidence might not be available
                      else:
                           logger.debug(f"Unexpected prediction result format for {model_name}: {prediction_result}")


                      if top_prediction_dest is not None:
                           predictions.append(top_prediction_dest)
                           confidences.append(float(top_confidence)) # Ensure float
                           valid_predictions += 1
                      else:
                           # Handle cases where no prediction is made or format is wrong
                           predictions.append(None) # Use None as a placeholder
                           confidences.append(0.0)
                           logger.debug(f"No valid prediction obtained for a row with {model_name}.")


                 except Exception as pred_e:
                      logger.warning(f"Prediction error for {model_name} on a row: {pred_e}", exc_info=True) # Log traceback for debug
                      predictions.append(None) # Placeholder on error
                      confidences.append(0.0)


            end_time = time.time()
            eval_duration = end_time - start_time
            inference_time_per_sample = eval_duration / len(test_df) if len(test_df) > 0 else 0

            # --- Calculate Metrics ---
            # Filter out None placeholders before calculating metrics
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            if not valid_indices:
                 logger.error(f"No valid predictions were made by {model_name}. Cannot calculate metrics.")
                 accuracy, balanced_acc, precision, recall, f1, avg_confidence, cm = 0, 0, 0, 0, 0, 0, np.array([])
            else:
                filtered_preds = [predictions[i] for i in valid_indices]
                filtered_truth = [ground_truth_labels[i] for i in valid_indices]
                filtered_confs = [confidences[i] for i in valid_indices]

                try:
                    accuracy = accuracy_score(filtered_truth, filtered_preds)
                    balanced_acc = balanced_accuracy_score(filtered_truth, filtered_preds)
                    precision = precision_score(filtered_truth, filtered_preds, average='weighted', zero_division=0)
                    recall = recall_score(filtered_truth, filtered_preds, average='weighted', zero_division=0)
                    f1 = f1_score(filtered_truth, filtered_preds, average='weighted', zero_division=0)
                    avg_confidence = np.mean(filtered_confs) if filtered_confs else 0.0
                    # Ensure labels for confusion matrix cover all unique true labels and predicted labels
                    all_observed_labels = sorted(list(set(filtered_truth) | set(filtered_preds)))
                    cm = confusion_matrix(filtered_truth, filtered_preds, labels=all_observed_labels)

                    logger.info(f"{model_name} evaluation completed ({valid_predictions}/{len(test_df)} valid predictions).")
                    logger.info(f"  Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
                    logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    logger.info(f"  Avg Confidence: {avg_confidence:.4f}, Avg Inference Time: {inference_time_per_sample:.6f} s/sample")

                    # Plot and save confusion matrix
                    self._plot_confusion_matrix(cm, model_name, labels=all_observed_labels)

                except Exception as metric_e:
                      logger.error(f"Error calculating metrics for {model_name}: {metric_e}", exc_info=True)
                      accuracy, balanced_acc, precision, recall, f1, avg_confidence, cm = 'Metric Error', 'Metric Error', 'Metric Error', 'Metric Error', 'Metric Error', 'Metric Error', None


            # Store results (including placeholders if metrics failed)
            self.results[model_name] = {
                'training_time': training_times.get(model_name),
                'model': model, # Store reference to loaded model if needed later
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc, # Added balanced accuracy
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_confidence': avg_confidence,
                'inference_time': inference_time_per_sample,
                'predictions_made': valid_predictions,
                'total_test_samples': len(test_df),
                'confusion_matrix': cm.tolist() if cm is not None else None # Store as list for JSON
            }

        logger.info("--- Model Evaluation Phase Completed ---")
        return self.results

    def _plot_confusion_matrix(self, cm, model_name, labels):
        """Confusion matrix görselleştirir ve kaydeder"""
        if cm is None or not hasattr(cm, 'shape'):
            logger.warning(f"Cannot plot confusion matrix for {model_name} due to missing/invalid matrix data.")
            return
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            plt.title(f'{model_name} Confusion Matrix (Test Set)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot to evaluation subdirectory
            safe_model_name = model_name.lower().replace(" ", "_").replace("*", "star")
            plot_path = f'evaluation/confusion_matrices/{safe_model_name}_confusion_matrix.png'
            plt.savefig(plot_path, dpi=150) # Lower dpi for potentially large matrices if needed
            plt.close() # Close the plot to free memory
            logger.info(f"{model_name} confusion matrix saved to: {plot_path}")

        except Exception as e:
            logger.error(f"Confusion matrix plotting/saving error for {model_name}: {e}", exc_info=True)

    def compare_models(self):
        """Modelleri karşılaştırır, sonuçları DataFrame'e dönüştürür ve grafikleri oluşturur"""
        logger.info("--- Comparing Model Performance ---")

        if not self.results:
            logger.warning("No evaluation results available to compare.")
            return None

        # Prepare data for DataFrame
        comparison_data = {
            'Model': [], 'Accuracy': [], 'Balanced Accuracy': [], 'Precision': [], 'Recall': [],
            'F1 Score': [], 'Avg Confidence': [], 'Training Time (s)': [], 'Inference Time (ms)': []
        }

        for model_name, result in self.results.items():
             comparison_data['Model'].append(model_name)
             # Append metrics, handling potential errors stored as strings
             comparison_data['Accuracy'].append(result.get('accuracy') if isinstance(result.get('accuracy'), (int, float)) else np.nan)
             comparison_data['Balanced Accuracy'].append(result.get('balanced_accuracy') if isinstance(result.get('balanced_accuracy'), (int, float)) else np.nan)
             comparison_data['Precision'].append(result.get('precision') if isinstance(result.get('precision'), (int, float)) else np.nan)
             comparison_data['Recall'].append(result.get('recall') if isinstance(result.get('recall'), (int, float)) else np.nan)
             comparison_data['F1 Score'].append(result.get('f1_score') if isinstance(result.get('f1_score'), (int, float)) else np.nan)
             comparison_data['Avg Confidence'].append(result.get('avg_confidence') if isinstance(result.get('avg_confidence'), (int, float)) else np.nan)
             comparison_data['Training Time (s)'].append(result.get('training_time') if isinstance(result.get('training_time'), (int, float)) else np.nan)
             # Convert inference time to ms
             inf_time = result.get('inference_time')
             comparison_data['Inference Time (ms)'].append(inf_time * 1000 if isinstance(inf_time, (int, float)) else np.nan)


        comparison_df = pd.DataFrame(comparison_data)
        # Drop rows where essential metrics might be NaN (e.g., due to training errors) before sorting/plotting
        comparison_df.dropna(subset=['Accuracy', 'Balanced Accuracy', 'F1 Score'], inplace=True)

        # Sort by a relevant metric, e.g., Balanced Accuracy
        comparison_df.sort_values(by='Balanced Accuracy', ascending=False, inplace=True)


        # Save comparison results to CSV in the evaluation directory
        os.makedirs('evaluation', exist_ok=True)
        csv_path = 'evaluation/model_comparison_results.csv'
        try:
            comparison_df.to_csv(csv_path, index=False)
            logger.info(f"Comparison results saved to: {csv_path}")
        except Exception as e:
             logger.error(f"Error saving comparison results to CSV: {e}", exc_info=True)

        # Generate comparison charts
        if not comparison_df.empty:
             try:
                  self._create_comparison_charts(comparison_df)
                  logger.info("Comparison charts created successfully.")
             except Exception as e:
                  logger.error(f"Error creating comparison charts: {e}", exc_info=True)
        else:
             logger.warning("Cannot generate comparison charts because the comparison DataFrame is empty after handling errors.")


        return comparison_df

    def _create_comparison_charts(self, df):
        """Karşılaştırma grafikleri oluşturur"""
        logger.info("Creating comparison charts...")
        os.makedirs('evaluation/charts', exist_ok=True) # Ensure charts directory exists

        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style

        # --- Plot 1: Performance Metrics (Acc, Bal Acc, F1) ---
        plt.figure(figsize=(12, 7))
        metrics_to_plot = ['Accuracy', 'Balanced Accuracy', 'F1 Score']
        metrics_df = df[['Model'] + metrics_to_plot].melt(
            id_vars=['Model'], var_name='Metric', value_name='Score'
        )
        # Use the sorted order of Models from the DataFrame
        model_order = df['Model'].tolist()
        sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette='viridis', order=model_order)
        plt.title('Model Performance Metrics (Higher is Better)')
        plt.ylim(0, 1.05) # Extend ylim slightly
        plt.ylabel('Score')
        plt.xlabel('') # Remove redundant x-label
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
        chart_path1 = 'evaluation/charts/performance_metrics_comparison.png'
        plt.savefig(chart_path1, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance metrics chart saved to: {chart_path1}")

        # --- Plot 2: Training and Inference Times ---
        plt.figure(figsize=(12, 7))
        # Training Time
        ax1 = plt.subplot(1, 2, 1)
        sns.barplot(x='Model', y='Training Time (s)', data=df, palette='magma', order=model_order, ax=ax1)
        ax1.set_title('Training Time (seconds)')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)

        # Inference Time
        ax2 = plt.subplot(1, 2, 2)
        sns.barplot(x='Model', y='Inference Time (ms)', data=df, palette='plasma', order=model_order, ax=ax2)
        ax2.set_title('Avg. Inference Time per Sample (ms)')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        chart_path2 = 'evaluation/charts/time_comparison.png'
        plt.savefig(chart_path2, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Time comparison chart saved to: {chart_path2}")

        # --- Plot 3: Radar Chart ---
        # (Optional but kept from original code - ensure library is installed if needed)
        try:
            self._create_radar_chart(df.copy()) # Pass copy to avoid modification issues
            chart_path3 = 'evaluation/charts/radar_chart_comparison.png' # Save inside the function
            logger.info(f"Radar chart saved to: {chart_path3}")
        except Exception as e:
            logger.error(f"Could not create radar chart: {e}", exc_info=True)

        # --- Plot 4: Performance vs. Inference Time ---
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='Inference Time (ms)', y='Balanced Accuracy', hue='Model', size='Training Time (s)',
                        sizes=(50, 400), data=df, palette='tab10')
        plt.title('Performance (Balanced Accuracy) vs. Inference Time')
        plt.xlabel('Avg. Inference Time per Sample (ms)')
        plt.ylabel('Balanced Accuracy')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
        chart_path4 = 'evaluation/charts/performance_vs_time.png'
        plt.savefig(chart_path4, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance vs. Time chart saved to: {chart_path4}")


    def _create_radar_chart(self, df):
        """Radar chart oluşturur (matplotlib)."""
        # Normalize data for radar chart (0-1 range typically)
        # Select relevant metrics for radar chart
        metrics_for_radar = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score']
        # Handle potential NaN values before normalization
        df_radar = df[['Model'] + metrics_for_radar].copy()
        df_radar.fillna(0, inplace=True) # Fill NaN with 0 for radar, or handle differently

        categories = metrics_for_radar
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Get color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for i, (index, row) in enumerate(df_radar.iterrows()):
            data = row[categories].values.flatten().tolist()
            data += data[:1]
            color = colors[i % len(colors)]
            ax.plot(angles, data, linewidth=2, linestyle='solid', label=row['Model'], color=color)
            ax.fill(angles, data, color=color, alpha=0.1)

        plt.xticks(angles[:-1], categories)
        ax.set_yticks(np.linspace(0, 1, 6)) # Adjust y-ticks based on expected range (0-1)
        ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0, 1, 6)])
        plt.ylim(0, 1.05) # Set limit slightly above 1
        ax.set_title('Model Performance Comparison (Radar Chart)', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)) # Adjust legend position

        chart_path = 'evaluation/charts/radar_chart_comparison.png'
        plt.tight_layout() # Adjust layout before saving
        plt.savefig(chart_path, dpi=150)
        plt.close(fig) # Close the specific figure


    def _create_html_report(self, df):
        """İnteraktif HTML raporu oluşturur (Görselleri içerir)"""
        logger.info("Creating HTML report...")
        os.makedirs('evaluation', exist_ok=True) # Ensure base directory exists

        # Ensure comparison charts and confusion matrices have been generated first
        comparison_chart_path_rel = 'charts/model_comparison_charts.png' # Relative path for HTML
        time_chart_path_rel = 'charts/time_comparison.png'
        radar_chart_path_rel = 'charts/radar_chart_comparison.png'
        perf_vs_time_chart_path_rel = 'charts/performance_vs_time.png'


        html_content = f"""
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <title>Model Karşılaştırma Raporu</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
                h1, h2, h3 {{ color: #005b96; border-bottom: 2px solid #6497b1; padding-bottom: 5px;}}
                h1 {{ text-align: center; }}
                table {{ border-collapse: collapse; width: 95%; margin: 20px auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #b3cde0; padding: 10px 12px; text-align: left; }}
                th {{ background-color: #6497b1; color: white; }}
                tr:nth-child(even) {{ background-color: #e0f7fa; }}
                tr:hover {{ background-color: #b3cde0; }}
                .chart-container {{ margin-top: 30px; text-align: center; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}}
                .model-section {{ margin: 40px auto; padding: 20px; border: 1px solid #b3cde0; border-radius: 8px; background-color: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 90%; }}
                img {{ max-width: 95%; height: auto; display: block; margin: 15px auto; border: 1px solid #ddd; border-radius: 4px; }}
                .footer {{ text-align: center; margin-top: 30px; font-size: 0.9em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Yapay Zeka Modelleri Karşılaştırma Raporu</h1>
            <p style="text-align: center;">Oluşturulma Tarihi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Performans Metrikleri Karşılaştırması</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Doğruluk</th>
                        <th>Dengeli Doğruluk</th>
                        <th>Kesinlik</th>
                        <th>Duyarlılık</th>
                        <th>F1 Skoru</th>
                        <th>Ort. Güven</th>
                        <th>Eğitim Süresi (s)</th>
                        <th>Çıkarım Süresi (ms)</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Add table rows, handling potential NaN values for display
        for _, row in df.iterrows():
             html_content += f"""
                    <tr>
                        <td>{row['Model']}</td>
                        <td>{row['Accuracy']:.4f}</td>
                        <td>{row['Balanced Accuracy']:.4f}</td>
                        <td>{row['Precision']:.4f}</td>
                        <td>{row['Recall']:.4f}</td>
                        <td>{row['F1 Score']:.4f}</td>
                        <td>{row['Avg Confidence']:.4f}</td>
                        <td>{row['Training Time (s)']:.2f}</td>
                        <td>{row['Inference Time (ms)']:.2f}</td>
                    </tr>
             """

        html_content += """
                </tbody>
            </table>

            <div class="chart-container">
                <h2>Karşılaştırma Grafikleri</h2>
                <h3>Performans Metrikleri</h3>
                <img src="{comparison_chart_path_rel}" alt="Model Performance Metrics Comparison">
                <h3>Zaman Karşılaştırması</h3>
                <img src="{time_chart_path_rel}" alt="Training and Inference Time Comparison">
                 <h3>Radar Chart</h3>
                <img src="{radar_chart_path_rel}" alt="Radar Chart Comparison">
                <h3>Performans vs. Zaman</h3>
                <img src="{perf_vs_time_chart_path_rel}" alt="Performance vs Inference Time Comparison">
           </div>

            <h2>Model Detayları ve Karmaşıklık Matrisleri</h2>
        """

        # Add section for each model's confusion matrix
        for model_name in df['Model']:
            safe_model_name = model_name.lower().replace(" ", "_").replace("*", "star")
            cm_path_rel = f'confusion_matrices/{safe_model_name}_confusion_matrix.png' # Relative path
            html_content += f"""
            <div class="model-section">
                <h3>{model_name}</h3>
                <p>Karmaşıklık Matrisi (Confusion Matrix):</p>
                <img src="{cm_path_rel}" alt="{model_name} Confusion Matrix">
            </div>
            """

        html_content += """
            <div class="footer">
                <p>Rapor ModelEvaluator sınıfı tarafından otomatik olarak oluşturulmuştur.</p>
            </div>
        </body>
        </html>
        """

        # Save HTML file
        report_path = 'evaluation/model_comparison_report.html'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report created successfully: {report_path}")
        except Exception as e:
            logger.error(f"Error creating HTML report: {e}", exc_info=True)


    def get_best_model(self, metric='balanced_accuracy'): # Default to balanced_accuracy
        """En iyi modeli belirli bir metriğe göre döndürür (DataFrame'den alır)"""
        logger.info(f"Finding best model based on '{metric}'...")
        comparison_df = self.compare_models() # Re-run comparison to get sorted DF if needed

        if comparison_df is None or comparison_df.empty:
            logger.warning("Cannot determine best model: No comparison results.")
            return None, None # Return None for both model name and instance

        # Map metric name if needed (e.g., F1 Score -> f1_score)
        metric_col_map = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'f1': 'F1 Score'
        }
        metric_col = metric_col_map.get(metric.lower(), metric) # Use provided name if not mapped

        if metric_col not in comparison_df.columns:
            logger.error(f"Metric '{metric_col}' not found in comparison results. Cannot determine best model.")
            return None, None

        # Sort by the desired metric, descending. Handle NaNs.
        best_model_row = comparison_df.sort_values(by=metric_col, ascending=False, na_position='last').iloc[0]
        best_model_name = best_model_row['Model']
        best_score = best_model_row[metric_col]

        # Get the corresponding trained model instance
        best_model_instance = self.trained_models.get(best_model_name)

        if best_model_instance is None:
             logger.warning(f"Best model according to metric '{metric_col}' is '{best_model_name}' with score {best_score:.4f}, but the trained instance was not found (possibly due to training error).")
             return best_model_name, None
        else:
             logger.info(f"Best model ({metric_col}): '{best_model_name}' (Score: {best_score:.4f})")
             return best_model_name, best_model_instance


# Example Usage (if run as a script)
if __name__ == "__main__":
    # Load your data
    DATA_FILE = 'models/synthetic_vacation_data.csv' # Adjust path if needed
    try:
        main_df = pd.read_csv(DATA_FILE)
        logger.info(f"Data loaded successfully from {DATA_FILE}")

        # Create evaluator instance
        evaluator = ModelEvaluator(model_save_dir='models/saved_models') # Specify save dir

        # Run the combined train and evaluation process
        evaluator.train_and_evaluate_all_models(main_df, test_size=0.3, random_state=123) # Use different state/size

        # Get the best model based on F1 score
        best_model_name, best_model_instance = evaluator.get_best_model(metric='f1_score')

        # You can now potentially use the best_model_instance for predictions
        if best_model_instance:
            logger.info(f"Retrieved best model instance: {best_model_name}")
            # Example prediction (requires appropriate input format)
            # sample_prefs = {'season': 'Summer', 'preferred_activity': 'Beach', 'budget': 1500, 'duration': 7}
            # try:
            #     prediction = best_model_instance.predict(sample_prefs)
            #     logger.info(f"Example prediction with {best_model_name}: {prediction}")
            # except Exception as e:
            #     logger.error(f"Error during example prediction: {e}")

    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_FILE}")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)