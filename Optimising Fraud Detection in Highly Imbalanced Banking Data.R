# =============================================================================
# Topic: Fraud Detection in Imbalanced Banking Data
# Research Question: Digital vs Demographic Features in Fraud Detection
# =============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(car)
library(lmtest)

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

data <- read.csv("Bank Account Fraud Dataset.csv") %>%
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

feature_stats <- data %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  group_by(name) %>%
  summarise(
    Mean = round(mean(value), 2),
    SD = round(sd(value), 2),
    Min = min(value),
    Max = max(value),
    Type = ifelse(name %in% c(
      "name_email_similarity", "velocity_24h", "session_length_in_minutes"
    ), "Digital", "Demographic")
  )

fraud_comparison <- data %>%
  mutate(Class = ifelse(fraud_bool == 1, "Fraud", "Legitimate")) %>%
  group_by(Class) %>%
  summarise(
    avg_session = mean(session_length_in_minutes),
    avg_credit = mean(credit_risk_score),
    avg_income = mean(income),
    n = n()
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
    c(session_length_in_minutes, credit_risk_score, name_email_similarity),
    names_to = "Feature", values_to = "Value"
  )

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
# 6. MODEL EVALUATION (ROC / AUC)
# =============================================================================

pred_digital <- predict(model_digital, test_data, type = "response")
pred_demo <- predict(model_demo, test_data, type = "response")
pred_rf <- predict(rf_model, test_data, type = "prob")[, 2]

roc_digital <- roc(test_data$fraud_bool, pred_digital)
roc_demo <- roc(test_data$fraud_bool, pred_demo)
roc_rf <- roc(test_data$fraud_bool, pred_rf)

performance_df <- tibble(
  Model = c("Digital", "Demographic", "Random Forest"),
  AUC = c(auc(roc_digital), auc(roc_demo), auc(roc_rf))
)

p4 <- ggplot(performance_df, aes(reorder(Model, AUC), AUC, fill = Model)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(AUC, 3)), hjust = -0.2) +
  labs(title = "AUC Comparison") +
  scale_fill_manual(values = c("#A23B72", "#2E86AB", "#2ECC71"))

# =============================================================================
# 7. HYPOTHESIS TEST
# =============================================================================

digital_imp <- importance_scores %>% filter(Type == "Digital") %>% pull(MeanDecreaseGini)
demo_imp <- importance_scores %>% filter(Type == "Demographic") %>% pull(MeanDecreaseGini)

t_test <- t.test(digital_imp, demo_imp, alternative = "greater")

avg_importance <- importance_scores %>%
  group_by(Type) %>%
  summarise(Avg_Importance = mean(MeanDecreaseGini))

# =============================================================================
# PRINT RESULTS
# =============================================================================

print(dataset_summary)
print(feature_stats)
print(fraud_comparison)
print(performance_df)
print(avg_importance)

print("Likelihood ratio test (Digital vs Full):"); print(lr_digital)
print("Likelihood ratio test (Demo vs Full):"); print(lr_demo)
print("VIF values:"); print(vif_values)
print("T-test Digital > Demographic importance:"); print(t_test)
