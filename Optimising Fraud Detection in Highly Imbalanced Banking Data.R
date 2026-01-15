# =============================================================================
# Topic: Fraud Detection in Imbalanced Banking Data
# Research Question: Digital vs Demographic Features in Fraud Detection
# =============================================================================

# Install tidyverse and other required packages
# install.packages(c("tidyverse", "caret", "randomForest", "pROC", "car", "lmtest"))

# Load the packages
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(car)
library(lmtest)

# =============================================================================
# 1. DATA PREPARATION
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

cat("Data loaded: ", nrow(data), " rows\n")
cat("Fraud rate: ", round(mean(data$fraud_bool == 1) * 100, 2), "%\n")

# =============================================================================
# 2. VISUALIZATIONS - DISPLAY IMMEDIATELY
# =============================================================================

theme_set(theme_minimal())

# PLOT 1: Class distribution
cat("\n=== DISPLAYING PLOT 1: Class Distribution ===\n")
p1 <- data %>%
  count(fraud_bool) %>%
  mutate(label = ifelse(fraud_bool == 1, "Fraud", "Legitimate")) %>%
  ggplot(aes(label, n, fill = label)) +
  geom_col() +
  geom_text(aes(label = paste0(round(n/sum(n)*100, 1), "%")), vjust = -0.5) +
  scale_fill_manual(values = c("#2E86AB", "#A23B72")) +
  labs(title = "Class Distribution", x = "", y = "Count") +
  theme(legend.position = "none")

print(p1)
readline("Press Enter for next plot...")

# Prepare data for Plot 2
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

# PLOT 2: Feature distributions
cat("\n=== DISPLAYING PLOT 2: Feature Distributions ===\n")
p2 <- ggplot(data_long, aes(Class, Value, fill = Class)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  scale_fill_manual(values = c("#2E86AB", "#A23B72")) +
  labs(title = "Feature Distributions", x = "", y = "") +
  theme(legend.position = "none")

print(p2)
readline("Press Enter for next plot...")

# =============================================================================
# 3. RANDOM FOREST TRAINING
# =============================================================================

set.seed(123)
train_index <- createDataPartition(data$fraud_bool, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

train_balanced <- upSample(
  x = train_data[, -1], y = train_data$fraud_bool,
  yname = "fraud_bool"
)

cat("\nTraining Random Forest...\n")
rf_model <- randomForest(fraud_bool ~ ., data = train_balanced,
                         ntree = 100, importance = TRUE)

# =============================================================================
# 4. FEATURE IMPORTANCE PLOT
# =============================================================================

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

# PLOT 3: Feature importance
cat("\n=== DISPLAYING PLOT 3: Feature Importance ===\n")
p3 <- importance_scores %>%
  ggplot(aes(reorder(Feature, MeanDecreaseGini), MeanDecreaseGini, fill = Type)) +
  geom_col() +
  coord_flip() +
  labs(title = "Feature Importance", x = "", y = "Mean Decrease Gini") +
  scale_fill_manual(values = c("Digital" = "#A23B72", "Demographic" = "#2E86AB")) +
  theme_minimal()

print(p3)
readline("Press Enter for next plot...")

# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================

# Use ACTUAL features from your dataset
digital_features <- c("name_email_similarity", "velocity_24h", 
                      "session_length_in_minutes", "device_os")
demographic_features <- c("income", "customer_age", "employment_status", 
                          "housing_status", "credit_risk_score")

# Train logistic models
cat("\nTraining Logistic Regression Models...\n")

# Digital features model
formula_digital <- as.formula(paste("fraud_bool ~", paste(digital_features, collapse = " + ")))
model_digital <- glm(formula_digital, data = train_data, family = "binomial")
pred_digital <- predict(model_digital, test_data, type = "response")
auc_digital <- auc(roc(test_data$fraud_bool, pred_digital))

# Demographic features model
formula_demo <- as.formula(paste("fraud_bool ~", paste(demographic_features, collapse = " + ")))
model_demo <- glm(formula_demo, data = train_data, family = "binomial")
pred_demo <- predict(model_demo, test_data, type = "response")
auc_demo <- auc(roc(test_data$fraud_bool, pred_demo))

# Random Forest predictions
pred_rf <- predict(rf_model, test_data, type = "prob")[, 2]
auc_rf <- auc(roc(test_data$fraud_bool, pred_rf))

# =============================================================================
# 6. MODEL PERFORMANCE PLOT
# =============================================================================

# Collect results
results <- data.frame(
  Model = c("Digital Features", "Demographic Features", "Random Forest"),
  AUC = c(auc_digital, auc_demo, auc_rf)
)

# PLOT 4: Model performance comparison
cat("\n=== DISPLAYING PLOT 4: Model Performance Comparison ===\n")
p4 <- ggplot(results, aes(x = reorder(Model, AUC), y = AUC, fill = Model)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = round(AUC, 3)), vjust = -0.5, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Model Performance Comparison", 
       subtitle = "Area Under ROC Curve (AUC)",
       x = "", y = "AUC Score") +
  ylim(0, 1) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p4)
readline("Press Enter for next plot...")

# =============================================================================
# 7. DIGITAL VS DEMOGRAPHIC IMPORTANCE PLOT
# =============================================================================

# Prepare data for Plot 5
imp_df <- importance_scores %>%
  filter(Type %in% c("Digital", "Demographic"))

# PLOT 5: Digital vs Demographic comparison
cat("\n=== DISPLAYING PLOT 5: Digital vs Demographic Features ===\n")
p5 <- ggplot(imp_df, 
             aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini, fill = Type)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Digital" = "tomato", "Demographic" = "steelblue")) +
  labs(title = "Digital vs Demographic Features",
       x = "", y = "Importance (Mean Decrease Gini)",
       fill = "Feature Type") +
  theme_minimal()

print(p5)

# =============================================================================
# 8. SAVE ALL PLOTS
# =============================================================================

cat("\n\nSaving all plots to files...\n")
ggsave("plot1_class_distribution.png", p1, width = 8, height = 6, dpi = 300)
ggsave("plot2_feature_distributions.png", p2, width = 10, height = 6, dpi = 300)
ggsave("plot3_feature_importance.png", p3, width = 8, height = 6, dpi = 300)
ggsave("plot4_model_performance.png", p4, width = 8, height = 6, dpi = 300)
ggsave("plot5_digital_vs_demographic.png", p5, width = 8, height = 6, dpi = 300)

cat("\n=== ALL PLOTS DISPLAYED AND SAVED ===\n")
cat("1. Class Distribution\n")
cat("2. Feature Distributions\n")
cat("3. Feature Importance\n")
cat("4. Model Performance Comparison\n")
cat("5. Digital vs Demographic Features\n\n")

cat("Model Performance Summary:\n")
cat("Digital Features AUC: ", round(auc_digital, 3), "\n")
cat("Demographic Features AUC: ", round(auc_demo, 3), "\n")
cat("Random Forest AUC: ", round(auc_rf, 3), "\n")

# Statistical test
digital_imp <- imp_df$MeanDecreaseGini[imp_df$Type == "Digital"]
demo_imp <- imp_df$MeanDecreaseGini[imp_df$Type == "Demographic"]

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
}

cat("\nPlots saved to:", getwd())
