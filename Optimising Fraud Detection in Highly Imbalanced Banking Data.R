# =============================================================================
# Topic: Fraud Detection in Imbalanced Banking Data
# Research Question: Digital vs Demographic Features in Fraud Detection
# =============================================================================

# Install tidyverse and other required packages
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("pROC")
install.packages("car")
install.packages("lmtest")

# Load the packages
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(car)
library(lmtest)

# =============================================================================
# 1. DATA PREPARATION WITH FULL FILE PATH
# =============================================================================

# Use the full path to your file
data <- read.csv("C:/Users/lenovo/Downloads/Bank Account Fraud Dataset.csv") %>%
  select(
    fraud_bool, name_email_similarity, device_os, velocity_24h,
    session_length_in_minutes, income, customer_age, employment_status,
    housing_status, credit_risk_score
  ) %>%
  mutate(
    fraud_bool = as.factor(fraud_bool),
    across(where(is.numeric), ~replace(., . == -1, NA)),
    device_os = as.factor(device_os),
    employment_status = as.factor(employment_status),
    housing_status = as.factor(housing_status)
  ) %>%
  na.omit()

# =============================================================================
# 2. DESCRIPTIVE STATS
# =============================================================================

dataset_summary <- tibble(
  Metric = c("Transactions", "Fraud", "Legitimate", "Fraud Rate", "Features"),
  Value = c(
    nrow(data),
    sum(data$fraud_bool == 1),
    sum(data$fraud_bool == 0),
    paste0(round(mean(data$fraud_bool == 1) * 100, 2), "%"),
    ncol(data) - 1
  )
)

# First, let's check what columns we actually have
cat("Columns in the dataset:\n")
print(names(data))
cat("\n")

# Feature statistics based on ACTUAL columns in the data
feature_stats <- data %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  reframe(
    Feature = name,
    Mean = round(mean(value), 2),
    SD = round(sd(value), 2),
    Min = min(value),
    Max = max(value),
    Type = ifelse(Feature %in% c(
      "name_email_similarity", "velocity_24h", "session_length_in_minutes"
    ), "Digital", "Demographic")
  )

# Fraud vs Legitimate comparison
fraud_comparison <- data %>%
  mutate(Class = ifelse(fraud_bool == 1, "Fraud", "Legitimate")) %>%
  group_by(Class) %>%
  summarise(
    avg_velocity_24h = mean(velocity_24h, na.rm = TRUE),
    avg_credit_score = mean(credit_risk_score, na.rm = TRUE),
    avg_income = mean(income, na.rm = TRUE),
    avg_age = mean(customer_age, na.rm = TRUE),
    avg_email_similarity = mean(name_email_similarity, na.rm = TRUE),
    n = n(),
    .groups = 'drop'
  )

# =============================================================================
# 3. VISUALIZATIONS
# =============================================================================

theme_set(theme_minimal())

# Class distribution
p1 <- data %>%
  count(fraud_bool) %>%
  mutate(label = ifelse(fraud_bool == 1, "Fraud", "Legitimate")) %>%
  ggplot(aes(label, n, fill = label)) +
  geom_col() +
  geom_text(aes(label = paste0(round(n/sum(n)*100, 1), "%")), vjust = -0.5) +
  scale_fill_manual(values = c("#2E86AB", "#A23B72")) +
  labs(title = "Class Distribution") +
  theme(legend.position = "none")

# Feature distributions
data_long <- data %>%
  mutate(Class = ifelse(fraud_bool == 1, "Fraud", "Legitimate")) %>%
  pivot_longer(
    c(velocity_24h, credit_risk_score, name_email_similarity, income),
    names_to = "Feature", values_to = "Value"
  ) %>%
  mutate(Feature = case_when(
    Feature == "velocity_24h" ~ "Transaction Velocity (24h)",
    Feature == "credit_risk_score" ~ "Credit Risk Score",
    Feature == "name_email_similarity" ~ "Name-Email Similarity",
    Feature == "income" ~ "Annual Income"
  ))

p2 <- ggplot(data_long, aes(Class, Value, fill = Class)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free") +
  scale_fill_manual(values = c("#2E86AB", "#A23B72")) +
  labs(title = "Feature Distributions") +
  theme(legend.position = "none")

# =============================================================================
# 4. HYPOTHESIS TESTING – LOGISTIC REGRESSION MODELS
# =============================================================================

model_data <- data %>%
  mutate(
    device_os = as.numeric(device_os),
    employment_status = as.numeric(employment_status),
    housing_status = as.numeric(housing_status)
  )

model_digital <- glm(
  fraud_bool ~ name_email_similarity + velocity_24h + session_length_in_minutes + device_os,
  data = model_data, family = "binomial"
)

model_demo <- glm(
  fraud_bool ~ income + customer_age + employment_status + housing_status + credit_risk_score,
  data = model_data, family = "binomial"
)

model_full <- glm(fraud_bool ~ ., data = model_data, family = "binomial")

# Likelihood ratio tests
lr_digital <- lrtest(model_digital, model_full)
lr_demo <- lrtest(model_demo, model_full)

# VIF for multicollinearity
vif_values <- vif(model_full)

# =============================================================================
# 5. RANDOM FOREST
# =============================================================================

set.seed(123)
train_index <- createDataPartition(data$fraud_bool, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

train_balanced <- upSample(
  x = train_data[, -1], y = train_data$fraud_bool,
  yname = "fraud_bool"
)

rf_model <- randomForest(fraud_bool ~ ., data = train_balanced,
                         ntree = 150, importance = TRUE)

importance_scores <- importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("Feature") %>%
  mutate(
    Type = ifelse(
      Feature %in% c("name_email_similarity","velocity_24h",
                     "session_length_in_minutes","device_os"),
      "Digital", "Demographic"
    )
  ) %>%
  arrange(desc(MeanDecreaseGini))

# Feature importance plot
p3 <- importance_scores %>%
  ggplot(aes(reorder(Feature, MeanDecreaseGini), MeanDecreaseGini, fill = Type)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance") +
  scale_fill_manual(values = c("Digital" = "#A23B72", "Demographic" = "#2E86AB"))
# =============================================================================
# 6. MODEL EVALUATION - SIMPLIFIED
# =============================================================================

cat("Starting Model Evaluation...\n")

# First, check what columns we actually have
cat("\nAvailable columns:\n")
print(names(data))

# Keep only columns that exist in the data
existing_cols <- names(data)

# Define features based on what exists
digital_features <- c("name_email_similarity", "days_since_request", 
                      "zip_count_4w", "payment_type")
demographic_features <- c("income", "customer_age", "prev_address_months_count", 
                          "current_address_months_count")

# Filter to only keep features that exist
digital_features <- digital_features[digital_features %in% existing_cols]
demographic_features <- demographic_features[demographic_features %in% existing_cols]

cat("\nDigital features found:", paste(digital_features, collapse = ", "), "\n")
cat("Demographic features found:", paste(demographic_features, collapse = ", "), "\n")

# Create train/test split
set.seed(123)
train_index <- createDataPartition(data$fraud_bool, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("\nTraining samples:", nrow(train_data), "\n")
cat("Test samples:", nrow(test_data), "\n")

# =============================================================================
# 6A. LOGISTIC REGRESSION MODELS
# =============================================================================

cat("\nTraining Logistic Regression Models...\n")

# Function to create formula from feature list
create_formula <- function(features) {
  if (length(features) == 0) {
    return(NULL)
  }
  as.formula(paste("fraud_bool ~", paste(features, collapse = " + ")))
}

# Create formulas
formula_digital <- create_formula(digital_features)
formula_demo <- create_formula(demographic_features)

# Train models only if we have features
if (!is.null(formula_digital)) {
  model_digital <- glm(formula_digital, data = train_data, family = "binomial")
  pred_digital <- predict(model_digital, test_data, type = "response")
  roc_digital <- roc(test_data$fraud_bool, pred_digital)
  auc_digital <- auc(roc_digital)
} else {
  auc_digital <- NA
  cat("No digital features available for model.\n")
}

if (!is.null(formula_demo)) {
  model_demo <- glm(formula_demo, data = train_data, family = "binomial")
  pred_demo <- predict(model_demo, test_data, type = "response")
  roc_demo <- roc(test_data$fraud_bool, pred_demo)
  auc_demo <- auc(roc_demo)
} else {
  auc_demo <- NA
  cat("No demographic features available for model.\n")
}

# =============================================================================
# 6B. FASTER RANDOM FOREST MODEL
# =============================================================================

cat("\nTraining Random Forest Model (Optimized)...\n")

# OPTION 1: Use smaller sample for faster training
# Take a smaller but balanced sample
set.seed(123)
sample_size <- 5000  # Adjust based on your computer speed

if (nrow(train_data) > sample_size) {
  # Sample equal numbers of fraud and legitimate cases
  fraud_indices <- which(train_data$fraud_bool == 1)
  legit_indices <- which(train_data$fraud_bool == 0)
  
  # Take up to half of sample_size from each class
  n_each <- min(sample_size %/% 2, length(fraud_indices), length(legit_indices))
  
  sampled_indices <- c(
    sample(fraud_indices, n_each),
    sample(legit_indices, n_each)
  )
  
  train_sample <- train_data[sampled_indices, ]
  cat("Using sampled data:", nrow(train_sample), "rows\n")
} else {
  train_sample <- train_data
  cat("Using full training data:", nrow(train_sample), "rows\n")
}

# OPTION 2: Use faster settings for Random Forest
rf_model <- randomForest(
  fraud_bool ~ .,
  data = train_sample,
  ntree = 50,          # Reduced from 100-150
  mtry = sqrt(ncol(train_sample) - 1),  # Default is good
  nodesize = 10,       # Larger nodes = faster
  maxnodes = 50,       # Limit tree size
  importance = TRUE,
  do.trace = TRUE      # Show progress
)

# Make predictions
pred_rf <- predict(rf_model, test_data, type = "prob")[, 2]
roc_rf <- roc(test_data$fraud_bool, pred_rf)
auc_rf <- auc(roc_rf)

cat("Random Forest trained successfully!\n")
cat("AUC:", round(auc_rf, 3), "\n")

# =============================================================================
# 6C 
# =============================================================================

library(ggplot2)

# 6C: MODEL PERFORMANCE RESULTS
cat("\n")
cat(rep("=", 60), sep = "")
cat("\nMODEL PERFORMANCE RESULTS\n")
cat(rep("=", 60), sep = "")
cat("\n\n")

# Collect results in a clean dataframe
results <- data.frame(
  Model = c(),
  AUC = c(),
  Features = c(),
  stringsAsFactors = FALSE
)

# Add available models
if (exists("auc_digital") && !is.na(auc_digital)) {
  results <- rbind(results, data.frame(
    Model = "Digital Features",
    AUC = round(auc_digital, 3),
    Features = paste(digital_features, collapse = ", "),
    stringsAsFactors = FALSE
  ))
}

if (exists("auc_demo") && !is.na(auc_demo)) {
  results <- rbind(results, data.frame(
    Model = "Demographic Features",
    AUC = round(auc_demo, 3),
    Features = paste(demographic_features, collapse = ", "),
    stringsAsFactors = FALSE
  ))
}

if (exists("auc_rf")) {
  results <- rbind(results, data.frame(
    Model = "Random Forest",
    AUC = round(auc_rf, 3),
    Features = "All features",
    stringsAsFactors = FALSE
  ))
}

# Display results
if (nrow(results) > 0) {
  print(results)
  cat("\n")
  
  # Simple bar plot
  results$Model <- factor(results$Model, levels = results$Model[order(results$AUC)])
  
  p1 <- ggplot(results, aes(x = Model, y = AUC, fill = Model)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = AUC), vjust = -0.5, size = 4) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      title = "Model Performance Comparison",
      subtitle = "Area Under ROC Curve (AUC)",
      y = "AUC Score"
    ) +
    ylim(0, max(results$AUC) * 1.1) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  print(p1)
  
  # Performance summary
  best_model <- results[which.max(results$AUC), ]
  cat("\nBest model:", best_model$Model, "(AUC =", best_model$AUC, ")\n")
}

# =============================================================================
# 7: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

cat("\n")
cat(rep("=", 60), sep = "")
cat("\nFEATURE IMPORTANCE ANALYSIS\n")
cat(rep("=", 60), sep = "")
cat("\n\n")

if (exists("rf_model") && !is.null(rf_model$importance)) {
  
  # Extract importance
  imp_df <- data.frame(
    Feature = rownames(rf_model$importance),
    Importance = rf_model$importance[, "MeanDecreaseGini"],
    stringsAsFactors = FALSE
  )
  
  # Categorize features
  imp_df$Type <- ifelse(
    imp_df$Feature %in% digital_features, "Digital",
    ifelse(imp_df$Feature %in% demographic_features, "Demographic", "Other")
  )
  
  # Sort by importance
  imp_df <- imp_df[order(-imp_df$Importance), ]
  
  # Show top 10 features
  cat("Top 10 Most Important Features:\n")
  print(head(imp_df, 10))
  cat("\n")
  
  # Feature importance plot
  imp_filtered <- imp_df[imp_df$Type %in% c("Digital", "Demographic"), ]
  
  if (nrow(imp_filtered) > 0) {
    p2 <- ggplot(imp_filtered, 
                 aes(x = reorder(Feature, Importance), y = Importance, fill = Type)) +
      geom_col() +
      coord_flip() +
      scale_fill_manual(values = c("Digital" = "tomato", "Demographic" = "steelblue")) +
      labs(
        title = "Feature Importance for Fraud Detection",
        x = "",
        y = "Importance (Mean Decrease Gini)"
      ) +
      theme_minimal()
    
    print(p2)
    
    # Statistical comparison
    digital_imp <- imp_filtered$Importance[imp_filtered$Type == "Digital"]
    demo_imp <- imp_filtered$Importance[imp_filtered$Type == "Demographic"]
    
    if (length(digital_imp) >= 2 && length(demo_imp) >= 2) {
      
      # Simple comparison
      mean_digital <- mean(digital_imp)
      mean_demo <- mean(demo_imp)
      
      cat("\nAverage Feature Importance:\n")
      cat("Digital features:   ", round(mean_digital, 3), "\n")
      cat("Demographic features:", round(mean_demo, 3), "\n")
      cat("Difference:         ", round(mean_digital - mean_demo, 3), "\n")
      
      # T-test
      t_test <- t.test(digital_imp, demo_imp, alternative = "greater")
      cat("\nT-test (Digital > Demographic):\n")
      cat("p-value:", round(t_test$p.value, 4), "\n")
      
      # Conclusion
      cat("\nRESEARCH CONCLUSION:\n")
      if (t_test$p.value < 0.05) {
        if (mean_digital > mean_demo) {
          cat("✓ Digital features are significantly more important for fraud detection.\n")
        } else {
          cat("✓ Demographic features are significantly more important for fraud detection.\n")
        }
      } else {
        cat("○ No significant difference between digital and demographic features.\n")
      }
    }
  }
}

cat("\n")
cat(rep("=", 60), sep = "")
cat("\nANALYSIS COMPLETE\n")
cat(rep("=", 60), sep = "")
cat("\n")

]