# =============================================================================
# Topic: Fraud Detection in Imbalanced Banking Data (patched)
# Research Question: Digital vs Demographic Features in Fraud Detection
# =============================================================================

# Install required packages if missing (uncomment to run)
# install.packages(c("tidyverse", "caret", "randomForest", "pROC", "readr"))

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(readr)

# -----------------------
# 0. CONFIG / PATH
# -----------------------
# Prefer a relative path in a project; fallback to interactive file chooser
data_path <- "data/Bank Account Fraud Dataset.csv"
if (!file.exists(data_path)) {
  if (interactive()) {
    message("Default data path not found. Please choose the CSV file interactively.")
    data_path <- file.choose()
  } else {
    stop("Data file not found at '", data_path, "'. Please provide the correct path or run interactively.")
  }
}

# -----------------------
# 1. LOAD & PREPARE DATA
# -----------------------
raw <- read_csv(data_path, show_col_types = FALSE)

# Required columns (edit this list if your column names differ)
required_cols <- c(
  "fraud_bool", "name_email_similarity", "device_os", "velocity_24h",
  "session_length_in_minutes", "income", "customer_age", "employment_status",
  "housing_status", "credit_risk_score"
)

missing_cols <- setdiff(required_cols, names(raw))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

data <- raw %>%
  select(all_of(required_cols)) %>%
  # Replace -1 sentinel with NA for numeric columns
  mutate(across(where(is.numeric), ~ replace(., . == -1, NA))) %>%
  # Coerce categorical variables to factors
  mutate(
    device_os = as.factor(device_os),
    employment_status = as.factor(employment_status),
    housing_status = as.factor(housing_status)
  ) %>%
  # Keep rows without NA in the selected features
  na.omit()

# Ensure fraud_bool is a factor with levels "0","1" and compute fraud rate explicitly
# If fraud_bool is numeric in the source, convert to character then factor
if (!is.factor(data$fraud_bool)) {
  data$fraud_bool <- as.factor(as.character(data$fraud_bool))
}
# Force levels to c("0","1") if both present (adjust if your positive label is different)
if (all(c("0", "1") %in% levels(data$fraud_bool))) {
  data$fraud_bool <- factor(data$fraud_bool, levels = c("0", "1"))
}

cat("Data loaded: ", nrow(data), " rows\n")
# Compute fraud rate by comparing to the "1" level
fraud_rate <- mean(as.character(data$fraud_bool) == "1")
cat("Fraud rate: ", round(fraud_rate * 100, 2), "%\n")

# -----------------------
# 2. VISUALIZATIONS
# -----------------------
theme_set(theme_minimal())

# PLOT 1: Class distribution
cat("\n=== DISPLAYING PLOT 1: Class Distribution ===\n")
p1 <- data %>%
  count(fraud_bool) %>%
  mutate(label = ifelse(as.character(fraud_bool) == "1", "Fraud", "Legitimate")) %>%
  ggplot(aes(label, n, fill = label)) +
  geom_col() +
  geom_text(aes(label = paste0(round(n / sum(n) * 100, 1), "%")), vjust = -0.5) +
  scale_fill_manual(values = c("Fraud" = "#A23B72", "Legitimate" = "#2E86AB")) +
  labs(title = "Class Distribution", x = "", y = "Count") +
  theme(legend.position = "none")

print(p1)
if (interactive()) readline("Press Enter for next plot...")

# Prepare data for Plot 2
data_long <- data %>%
  mutate(Class = ifelse(as.character(fraud_bool) == "1", "Fraud", "Legitimate")) %>%
  pivot_longer(
    c(velocity_24h, credit_risk_score, name_email_similarity, income),
    names_to = "Feature", values_to = "Value"
  ) %>%
  mutate(Feature = case_when(
    Feature == "velocity_24h" ~ "Transaction Velocity (24h)",
    Feature == "credit_risk_score" ~ "Credit Risk Score",
    Feature == "name_email_similarity" ~ "Name-Email Similarity",
    Feature == "income" ~ "Annual Income",
    TRUE ~ Feature
  ))

# PLOT 2: Feature distributions
cat("\n=== DISPLAYING PLOT 2: Feature Distributions ===\n")
p2 <- ggplot(data_long, aes(Class, Value, fill = Class)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("Fraud" = "#A23B72", "Legitimate" = "#2E86AB")) +
  labs(title = "Feature Distributions", x = "", y = "") +
  theme(legend.position = "none")

print(p2)
if (interactive()) readline("Press Enter for next plot...")

# -----------------------
# 3. RANDOM FOREST TRAINING
# -----------------------
set.seed(123)
train_index <- createDataPartition(data$fraud_bool, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Use upSample with explicit x and y (safer than dropping first column assumption)
set.seed(456) # reproducible upsampling
train_balanced <- upSample(
  x = train_data %>% select(-fraud_bool),
  y = train_data$fraud_bool,
  yname = "fraud_bool"
)

cat("\nTraining Random Forest...\n")
rf_model <- randomForest(fraud_bool ~ ., data = train_balanced,
                         ntree = 100, importance = TRUE)

# -----------------------
# 4. FEATURE IMPORTANCE
# -----------------------
imp_mat <- importance(rf_model)

# Determine which importance column to use
importance_col <- NULL
if (is.matrix(imp_mat) || is.data.frame(imp_mat)) {
  if ("MeanDecreaseGini" %in% colnames(imp_mat)) importance_col <- "MeanDecreaseGini"
  else if ("MeanDecreaseAccuracy" %in% colnames(imp_mat)) importance_col <- "MeanDecreaseAccuracy"
  else {
    # if importance is a named vector
    importance_col <- NULL
  }
}

if (is.null(importance_col)) {
  # fallback: attempt to coerce importance to a single numeric vector
  imp_df <- tibble(Feature = rownames(imp_mat), Importance = as.numeric(imp_mat[, 1]))
  colnames(imp_df)[2] <- "Importance"
  warning("Could not find MeanDecreaseGini or MeanDecreaseAccuracy; using first column of importance() output.")
} else {
  imp_df <- as.data.frame(imp_mat) %>%
    rownames_to_column("Feature") %>%
    rename(Importance = all_of(importance_col))
}

# Tag digital vs demographic (edit lists if your features differ)
digital_set <- c("name_email_similarity", "velocity_24h", "session_length_in_minutes", "device_os")
imp_df <- imp_df %>%
  mutate(Type = ifelse(Feature %in% digital_set, "Digital", "Demographic")) %>%
  arrange(desc(Importance))

# PLOT 3: Feature importance
cat("\n=== DISPLAYING PLOT 3: Feature Importance ===\n")
p3 <- imp_df %>%
  ggplot(aes(reorder(Feature, Importance), Importance, fill = Type)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance", x = "", y = ifelse(is.null(importance_col), "Importance", importance_col)) +
  scale_fill_manual(values = c("Digital" = "#A23B72", "Demographic" = "#2E86AB")) +
  theme_minimal()

print(p3)
if (interactive()) readline("Press Enter for next plot...")

# -----------------------
# 5. MODEL EVALUATION
# -----------------------
# Define feature groups (ensure these columns exist)
digital_features <- c("name_email_similarity", "velocity_24h", "session_length_in_minutes", "device_os")
demographic_features <- c("income", "customer_age", "employment_status", "housing_status", "credit_risk_score")

# Helper to safely compute AUC with explicit positive class = "1"
safe_auc <- function(labels_factor, preds) {
  # labels_factor should be factor with levels that include "0" and "1"
  if (!is.factor(labels_factor)) labels_factor <- as.factor(as.character(labels_factor))
  roc_obj <- tryCatch(
    roc(response = labels_factor, predictor = preds, levels = c("0", "1"), direction = "<", quiet = TRUE),
    error = function(e) NULL
  )
  if (is.null(roc_obj)) return(NA_real_)
  return(as.numeric(auc(roc_obj)))
}

cat("\nTraining Logistic Regression Models...\n")

# Digital features model
formula_digital <- as.formula(paste("fraud_bool ~", paste(digital_features, collapse = " + ")))
model_digital <- glm(formula_digital, data = train_data, family = "binomial")
pred_digital <- predict(model_digital, test_data, type = "response")
auc_digital <- safe_auc(test_data$fraud_bool, pred_digital)

# Demographic features model
formula_demo <- as.formula(paste("fraud_bool ~", paste(demographic_features, collapse = " + ")))
model_demo <- glm(formula_demo, data = train_data, family = "binomial")
pred_demo <- predict(model_demo, test_data, type = "response")
auc_demo <- safe_auc(test_data$fraud_bool, pred_demo)

# Random Forest predictions - pick probability for class "1"
pred_rf_probs <- predict(rf_model, test_data, type = "prob")
if ("1" %in% colnames(pred_rf_probs)) {
  pred_rf <- pred_rf_probs[, "1"]
} else {
  # fallback to second column if named differently
  pred_rf <- pred_rf_probs[, ncol(pred_rf_probs)]
}
auc_rf <- safe_auc(test_data$fraud_bool, pred_rf)

# -----------------------
# 6. MODEL PERFORMANCE PLOT
# -----------------------
results <- data.frame(
  Model = c("Digital Features", "Demographic Features", "Random Forest"),
  AUC = c(auc_digital, auc_demo, auc_rf)
)

cat("\n=== DISPLAYING PLOT 4: Model Performance Comparison ===\n")
p4 <- ggplot(results, aes(x = reorder(Model, AUC), y = AUC, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(AUC, 3)), vjust = -0.5, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Comparison", subtitle = "Area Under ROC Curve (AUC)", x = "", y = "AUC Score") +
  ylim(0, 1) +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

print(p4)
if (interactive()) readline("Press Enter for next plot...")

# -----------------------
# 7. DIGITAL VS DEMOGRAPHIC IMPORTANCE PLOT
# -----------------------
imp_plot_df <- imp_df %>% filter(Type %in% c("Digital", "Demographic"))

cat("\n=== DISPLAYING PLOT 5: Digital vs Demographic Features ===\n")
p5 <- ggplot(imp_plot_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Type)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Digital" = "tomato", "Demographic" = "steelblue")) +
  labs(title = "Digital vs Demographic Features", x = "", y = "Importance", fill = "Feature Type") +
  theme_minimal()

print(p5)

# -----------------------
# 8. SAVE ALL PLOTS
# -----------------------
plots_dir <- "plots"
if (!dir.exists(plots_dir)) dir.create(plots_dir)

cat("\nSaving all plots to files in:", plots_dir, "\n")
ggsave(file.path(plots_dir, "plot1_class_distribution.png"), p1, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "plot2_feature_distributions.png"), p2, width = 10, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "plot3_feature_importance.png"), p3, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "plot4_model_performance.png"), p4, width = 8, height = 6, dpi = 300)
ggsave(file.path(plots_dir, "plot5_digital_vs_demographic.png"), p5, width = 8, height = 6, dpi = 300)

cat("\n=== ALL PLOTS DISPLAYED AND SAVED ===\n")
cat("Model Performance Summary:\n")
cat("Digital Features AUC: ", ifelse(is.na(auc_digital), "NA", round(auc_digital, 3)), "\n")
cat("Demographic Features AUC: ", ifelse(is.na(auc_demo), "NA", round(auc_demo, 3)), "\n")
cat("Random Forest AUC: ", ifelse(is.na(auc_rf), "NA", round(auc_rf, 3)), "\n")

# -----------------------
# 9. STATISTICAL TEST (importance)
# -----------------------
digital_imp <- imp_plot_df$Importance[imp_plot_df$Type == "Digital"]
demo_imp <- imp_plot_df$Importance[imp_plot_df$Type == "Demographic"]

if (length(digital_imp) >= 2 && length(demo_imp) >= 2) {
  t_test <- t.test(digital_imp, demo_imp, alternative = "two.sided")
  cat("\nT-test (Digital vs Demographic importance):\n")
  cat("p-value: ", round(t_test$p.value, 4), "\n")

  cat("\nRESEARCH CONCLUSION:\n")
  if (t_test$p.value < 0.05) {
    if (mean(digital_imp) > mean(demo_imp)) {
      cat("✓ Digital features are SIGNIFICANTLY more important (p < 0.05)\n")
    } else {
      cat("✓ Demographic features are SIGNIFICANTLY more important (p < 0.05)\n")
    }
  } else {
    cat("○ No significant difference between feature types (p =", round(t_test$p.value, 3), ")\n")
  }
} else {
  cat("\nNot enough features in one of the groups to run a t-test reliably.\n")
}

cat("\nPlots saved to:", normalizePath(plots_dir), "\n")
