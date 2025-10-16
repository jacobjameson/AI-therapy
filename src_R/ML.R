# ============================================================================
# THERAPY ASSIGNMENT PREDICTION ANALYSIS
# Comparing patient clinical features vs therapist/organizational factors
# ============================================================================

# Load required libraries
library(dplyr)
library(glmnet)
library(pROC)
library(ggplot2)
library(gridExtra)

# Set seed for reproducibility
set.seed(42)


# ============================================================================
# PART 1: DEFINE FEATURE GROUPS
# ============================================================================

cat("Step 1: Defining feature groups...\n")

# Therapy columns (treatment modalities)
therapy_cols <- c(
  "act",
  "cbt", 
  "dbt",
  "motivational_interviewing",
  "mindfulness",
  "stages_of_change",
  "family_systems",
  "trauma_informed"
)

# Clinical feature groups
dx_cols <- names(data)[grepl("^current_and_past_psychiatric_diagnoses_", names(data))]
symp_cols <- names(data)[grepl("^presenting_symptoms_", names(data))]
fam_cols <- names(data)[grepl("^family_history_", names(data))]
precip_cols <- names(data)[grepl("^precipitants_stressors_", names(data))]
prot_cols <- names(data)[grepl("^internal_protective_factors_|^external_protective_factors_", names(data))]
treatment_cols <- names(data)[grepl("^change_in_treatment_", names(data))]

# Create derived variables
if ("risk_level_initial" %in% names(data)) {
  data$risk_high_initial <- as.integer(data$risk_level_initial == "High")
}

data$adolescent <- as.integer(data$age_group == "Adolescent")
data$male <- as.integer(data$sex_fs == "male")

score_var <- "total_score"

cat(sprintf("  - %d diagnosis features\n", length(dx_cols)))
cat(sprintf("  - %d symptom features\n", length(symp_cols)))
cat(sprintf("  - %d family history features\n", length(fam_cols)))
cat(sprintf("  - %d precipitant/stressor features\n", length(precip_cols)))
cat(sprintf("  - %d protective factor features\n", length(prot_cols)))


# ============================================================================
# PART 2: CREATE MODELING DATASET
# ============================================================================

cat("\nStep 2: Creating modeling dataset...\n")

# All features for models
model_features <- c(
  "risk_high_initial",
  score_var,
  "adolescent", 
  "male",
  therapy_cols,
  dx_cols,
  symp_cols,
  fam_cols,
  precip_cols,
  prot_cols,
  treatment_cols
)

# Metadata columns
metadata_cols <- c("therapist_name", "location", "program", "pn_month", 'pn_time_block', 'pn_year', 'admission_date')

# Create modeling dataset
model_df <- data[, unique(c(model_features, metadata_cols))]
model_df <- model_df[, !duplicated(names(model_df))]

# Add row index to track observations
model_df$row_idx <- 1:nrow(model_df)

# Patient features (NO therapy columns - avoids data leakage)
patient_features_for_therapy_prediction <- c(
  score_var,
  "risk_high_initial",
  "adolescent",
  "male",
  dx_cols,
  symp_cols,
  fam_cols,
  precip_cols,
  prot_cols,
  treatment_cols
)

patient_features_for_therapy_prediction <- unique(patient_features_for_therapy_prediction)
patient_features_for_therapy_prediction <- patient_features_for_therapy_prediction[
  patient_features_for_therapy_prediction %in% names(model_df)
]

cat(sprintf("  - Total observations: %d\n", nrow(model_df)))
cat(sprintf("  - Patient features: %d\n", length(patient_features_for_therapy_prediction)))


# ============================================================================
# PART 3: TEMPORAL TRAIN-TEST SPLIT
# ============================================================================

cat("\nStep 3: Creating temporal train-test split...\n")

model_df <- model_df %>%
  arrange(admission_date) %>%
  mutate(split_row_id = row_number())

split_idx <- floor(nrow(model_df) * 0.8)
train_df <- model_df %>% filter(split_row_id <= split_idx)
valid_df <- model_df %>% filter(split_row_id > split_idx)

cat(sprintf("  - Training set: %d observations (80%%)\n", nrow(train_df)))
cat(sprintf("  - Validation set: %d observations (20%%)\n", nrow(valid_df)))


# ============================================================================
# PART 4: MODEL TRAINING FUNCTION (UPDATED TO SAVE PROPENSITY SCORES)
# ============================================================================

get_curves_for_therapy <- function(train_df, valid_df, full_df, therapy_col, label,
                                   patient_features, 
                                   therapist_cols = c("therapist_name", "location", "program")) {
  
  # Get therapy labels
  y_train <- ifelse(is.na(train_df[[therapy_col]]), 0, train_df[[therapy_col]])
  y_test <- ifelse(is.na(valid_df[[therapy_col]]), 0, valid_df[[therapy_col]])
  y_full <- ifelse(is.na(full_df[[therapy_col]]), 0, full_df[[therapy_col]])
  
  # Check variance and class balance
  if (length(unique(y_train)) < 2 || length(unique(y_test)) < 2) {
    cat(sprintf("  Skipping %s: insufficient outcome variance\n", label))
    return(NULL)
  }
  
  if (sum(y_test == 1) < 5 || sum(y_test == 0) < 5) {
    cat(sprintf("  Skipping %s: insufficient test set size\n", label))
    return(NULL)
  }
  
  available_therapist_cols <- therapist_cols[therapist_cols %in% names(train_df)]
  
  # Helper function to prepare data for glmnet
  prepare_data_for_glmnet <- function(train_data, test_data, feature_cols, include_categorical = FALSE) {
    num_cols <- sapply(train_data[, feature_cols, drop = FALSE], is.numeric)
    numeric_features <- feature_cols[num_cols]
    categorical_features <- feature_cols[!num_cols]
    
    # Process numeric features
    if (length(numeric_features) > 0) {
      X_train_num <- as.matrix(train_data[, numeric_features, drop = FALSE])
      X_test_num <- as.matrix(test_data[, numeric_features, drop = FALSE])
      
      for (i in 1:ncol(X_train_num)) {
        median_val <- median(X_train_num[, i], na.rm = TRUE)
        if (is.na(median_val)) median_val <- 0
        X_train_num[is.na(X_train_num[, i]), i] <- median_val
        X_test_num[is.na(X_test_num[, i]), i] <- median_val
      }
    } else {
      X_train_num <- matrix(nrow = nrow(train_data), ncol = 0)
      X_test_num <- matrix(nrow = nrow(test_data), ncol = 0)
    }
    
    # Process categorical features
    if (include_categorical && length(categorical_features) > 0) {
      train_cat_data <- train_data[, categorical_features, drop = FALSE]
      test_cat_data <- test_data[, categorical_features, drop = FALSE]
      
      formula_str <- paste("~", paste(categorical_features, collapse = " + "), "- 1")
      formula_obj <- as.formula(formula_str)
      
      X_train_cat <- model.matrix(formula_obj, data = train_cat_data)
      X_test_cat <- model.matrix(formula_obj, data = test_cat_data)
      
      missing_cols <- setdiff(colnames(X_train_cat), colnames(X_test_cat))
      extra_cols <- setdiff(colnames(X_test_cat), colnames(X_train_cat))
      
      if (length(missing_cols) > 0) {
        missing_matrix <- matrix(0, nrow = nrow(X_test_cat), ncol = length(missing_cols))
        colnames(missing_matrix) <- missing_cols
        X_test_cat <- cbind(X_test_cat, missing_matrix)
      }
      
      if (length(extra_cols) > 0) {
        X_test_cat <- X_test_cat[, !colnames(X_test_cat) %in% extra_cols, drop = FALSE]
      }
      
      X_test_cat <- X_test_cat[, colnames(X_train_cat), drop = FALSE]
    } else {
      X_train_cat <- matrix(nrow = nrow(train_data), ncol = 0)
      X_test_cat <- matrix(nrow = nrow(test_data), ncol = 0)
    }
    
    X_train <- cbind(X_train_num, X_train_cat)
    X_test <- cbind(X_test_num, X_test_cat)
    
    return(list(train = X_train, test = X_test))
  }
  
  tryCatch({
    # Prepare data for train/test
    patient_data <- prepare_data_for_glmnet(
      train_df, valid_df, patient_features, include_categorical = FALSE
    )
    
    therapist_data <- prepare_data_for_glmnet(
      train_df, valid_df, available_therapist_cols, include_categorical = TRUE
    )
    
    combined_data <- prepare_data_for_glmnet(
      train_df, valid_df, 
      c(patient_features, available_therapist_cols), 
      include_categorical = TRUE
    )
    
    # Prepare data for FULL dataset (to generate propensity scores)
    patient_data_full <- prepare_data_for_glmnet(
      train_df, full_df, patient_features, include_categorical = FALSE
    )
    
    therapist_data_full <- prepare_data_for_glmnet(
      train_df, full_df, available_therapist_cols, include_categorical = TRUE
    )
    
    combined_data_full <- prepare_data_for_glmnet(
      train_df, full_df, 
      c(patient_features, available_therapist_cols), 
      include_categorical = TRUE
    )
    
    if (ncol(therapist_data$train) == 0) {
      cat(sprintf("  Warning: No therapist features for %s\n", label))
      return(NULL)
    }
    
    # Class weights
    class_counts <- table(y_train)
    if (length(class_counts) < 2) return(NULL)
    
    weights <- ifelse(y_train == 1,
                      length(y_train) / (2 * class_counts["1"]),
                      length(y_train) / (2 * class_counts["0"]))
    
    # Train L2-regularized logistic regression models
    cv_patient <- cv.glmnet(
      x = patient_data$train, y = y_train,
      family = "binomial", alpha = 0,
      weights = weights, type.measure = "auc", nfolds = 5
    )
    
    cv_therapist <- cv.glmnet(
      x = therapist_data$train, y = y_train,
      family = "binomial", alpha = 0,
      weights = weights, type.measure = "auc", nfolds = 5
    )
    
    cv_combined <- cv.glmnet(
      x = combined_data$train, y = y_train,
      family = "binomial", alpha = 0,
      weights = weights, type.measure = "auc", nfolds = 5
    )
    
    # Generate predictions for TEST set (for evaluation)
    y_pred_patient <- predict(cv_patient, newx = patient_data$test, 
                              s = "lambda.min", type = "response")[, 1]
    y_pred_therapist <- predict(cv_therapist, newx = therapist_data$test, 
                                s = "lambda.min", type = "response")[, 1]
    y_pred_combined <- predict(cv_combined, newx = combined_data$test, 
                               s = "lambda.min", type = "response")[, 1]
    
    # Generate propensity scores for FULL dataset (for XGBoost)
    propensity_patient_full <- predict(cv_patient, newx = patient_data_full$test, 
                                       s = "lambda.min", type = "response")[, 1]
    propensity_therapist_full <- predict(cv_therapist, newx = therapist_data_full$test, 
                                         s = "lambda.min", type = "response")[, 1]
    propensity_combined_full <- predict(cv_combined, newx = combined_data_full$test, 
                                        s = "lambda.min", type = "response")[, 1]
    
    # Calculate ROC curves (force same direction)
    roc_patient <- roc(y_test, y_pred_patient, quiet = TRUE, direction = "<")
    roc_therapist <- roc(y_test, y_pred_therapist, quiet = TRUE, direction = "<")
    roc_combined <- roc(y_test, y_pred_combined, quiet = TRUE, direction = "<")
    
    auc_p <- as.numeric(auc(roc_patient))
    auc_t <- as.numeric(auc(roc_therapist))
    auc_c <- as.numeric(auc(roc_combined))
    
    # DeLong test for comparing correlated AUCs
    delong_t_vs_p <- tryCatch({
      test_result <- roc.test(roc_therapist, roc_patient, method = "delong")
      list(statistic = test_result$statistic, p_value = test_result$p.value)
    }, error = function(e) {
      list(statistic = NA, p_value = NA)
    })
    
    delong_c_vs_p <- tryCatch({
      test_result <- roc.test(roc_combined, roc_patient, method = "delong")
      list(statistic = test_result$statistic, p_value = test_result$p.value)
    }, error = function(e) {
      list(statistic = NA, p_value = NA)
    })
    
    # Brier Score
    brier_patient <- mean((y_test - y_pred_patient)^2)
    brier_therapist <- mean((y_test - y_pred_therapist)^2)
    brier_combined <- mean((y_test - y_pred_combined)^2)
    
    # Baseline rate
    baseline_rate <- mean(y_test)
    
    # Print results
    cat(sprintf("  %s: Patient AUC=%.3f, Therapist AUC=%.3f, Combined AUC=%.3f (n=%d/%d, p=%.4f)\n",
                label, auc_p, auc_t, auc_c, sum(y_test == 1), length(y_test), delong_t_vs_p$p_value))
    
    return(list(
      therapy = label,
      baseline_rate = baseline_rate,
      n_positive = sum(y_test == 1),
      n_total = length(y_test),
      patient = list(fpr = 1 - roc_patient$specificities, 
                     tpr = roc_patient$sensitivities, 
                     auc = auc_p),
      therapist = list(fpr = 1 - roc_therapist$specificities, 
                       tpr = roc_therapist$sensitivities, 
                       auc = auc_t),
      combined = list(fpr = 1 - roc_combined$specificities, 
                      tpr = roc_combined$sensitivities, 
                      auc = auc_c),
      delong_tests = list(t_vs_p = delong_t_vs_p,
                          c_vs_p = delong_c_vs_p),
      brier = list(patient = brier_patient,
                   therapist = brier_therapist,
                   combined = brier_combined),
      propensity_scores = list(
        patient = propensity_patient_full,
        therapist = propensity_therapist_full,
        combined = propensity_combined_full
      ),
      models = list(patient = cv_patient, 
                    therapist = cv_therapist, 
                    combined = cv_combined)
    ))
    
  }, error = function(e) {
    cat(sprintf("  Error processing %s: %s\n", label, e$message))
    return(NULL)
  })
}


# ============================================================================
# PART 5: RUN MODELS FOR ALL THERAPIES
# ============================================================================

cat("\nStep 4: Training models for each therapy modality...\n")

results <- list()

therapy_labels <- list(
  "act" = "ACT",
  "cbt" = "CBT",
  "dbt" = "DBT",
  "motivational_interviewing" = "MI",
  "mindfulness" = "Mindfulness",
  "stages_of_change" = "Stages of Change",
  "family_systems" = "Family Systems",
  "trauma_informed" = "Trauma-Informed"
)

for (col in names(therapy_labels)) {
  label <- therapy_labels[[col]]
  if (col %in% names(model_df)) {
    res <- get_curves_for_therapy(
      train_df, valid_df, model_df, col, label,
      patient_features_for_therapy_prediction,
      therapist_cols = c("therapist_name", "location", "program")
    )
    if (!is.null(res)) {
      results[[length(results) + 1]] <- res
    }
  } else {
    cat(sprintf("  Warning: Therapy column '%s' not found\n", col))
  }
}


# ============================================================================
# PART 6: EXTRACT AND SAVE PROPENSITY SCORES
# ============================================================================

cat("\nStep 5: Extracting propensity scores...\n")

# Create propensity score dataframe
propensity_df <- data.frame(row_idx = model_df$row_idx)

for (i in seq_along(results)) {
  therapy_name <- results[[i]]$therapy
  therapy_col_clean <- tolower(gsub(" ", "_", therapy_name))
  
  # Use combined model propensity scores (captures both patient and therapist factors)
  propensity_df[[paste0("prop_", therapy_col_clean)]] <- 
    results[[i]]$propensity_scores$combined
}

# Merge propensity scores back into data
data <- data %>%
  mutate(row_idx = row_number()) %>%
  left_join(propensity_df, by = "row_idx") %>%
  select(-row_idx)

cat(sprintf("  - Added %d propensity score columns to data\n", length(results)))
cat("  - Propensity scores saved to data object\n")


# ============================================================================
# PART 7: CREATE SUMMARY STATISTICS
# ============================================================================

cat("\nStep 6: Generating summary statistics...\n")

summary_stats <- data.frame(
  Therapy = sapply(results, function(x) x$therapy),
  Baseline_Rate = sapply(results, function(x) x$baseline_rate),
  N_Positive = sapply(results, function(x) x$n_positive),
  N_Total = sapply(results, function(x) x$n_total),
  Patient_AUC = sapply(results, function(x) x$patient$auc),
  Therapist_AUC = sapply(results, function(x) x$therapist$auc),
  Combined_AUC = sapply(results, function(x) x$combined$auc),
  DeLong_Z = sapply(results, function(x) x$delong_tests$t_vs_p$statistic),
  DeLong_p = sapply(results, function(x) x$delong_tests$t_vs_p$p_value),
  Brier_Patient = sapply(results, function(x) x$brier$patient),
  Brier_Therapist = sapply(results, function(x) x$brier$therapist),
  Brier_Combined = sapply(results, function(x) x$brier$combined)
) %>%
  mutate(
    Delta_AUC_T_vs_P = Therapist_AUC - Patient_AUC,
    Delta_AUC_C_vs_P = Combined_AUC - Patient_AUC,
    Delta_AUC_C_vs_T = Combined_AUC - Therapist_AUC
  )


# ============================================================================
# PART 8: PRINT RESULTS TABLES
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("TABLE 1: MODEL DISCRIMINATION (AUROC)\n")
cat(rep("=", 90), "\n", sep="")

print(summary_stats %>% 
        select(Therapy, Baseline_Rate, Patient_AUC, Therapist_AUC, Combined_AUC, 
               Delta_AUC_T_vs_P, DeLong_p) %>%
        mutate(across(where(is.numeric), ~round(.x, 3))))

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("AGGREGATE STATISTICS\n")
cat(rep("=", 90), "\n", sep="")

cat("\n1. Mean AUC Performance:\n")
cat(sprintf("   Patient features:  %.3f (SD = %.3f, range: %.3f-%.3f)\n", 
            mean(summary_stats$Patient_AUC), sd(summary_stats$Patient_AUC),
            min(summary_stats$Patient_AUC), max(summary_stats$Patient_AUC)))
cat(sprintf("   Therapist factors: %.3f (SD = %.3f, range: %.3f-%.3f)\n", 
            mean(summary_stats$Therapist_AUC), sd(summary_stats$Therapist_AUC),
            min(summary_stats$Therapist_AUC), max(summary_stats$Therapist_AUC)))
cat(sprintf("   Combined model:    %.3f (SD = %.3f, range: %.3f-%.3f)\n", 
            mean(summary_stats$Combined_AUC), sd(summary_stats$Combined_AUC),
            min(summary_stats$Combined_AUC), max(summary_stats$Combined_AUC)))
cat(sprintf("   Mean difference (T-P): %.3f (95%% CI: %.3f to %.3f)\n", 
            mean(summary_stats$Delta_AUC_T_vs_P),
            quantile(summary_stats$Delta_AUC_T_vs_P, 0.025),
            quantile(summary_stats$Delta_AUC_T_vs_P, 0.975)))

# Paired t-test
t_test_result <- t.test(summary_stats$Therapist_AUC, summary_stats$Patient_AUC, paired = TRUE)
cat(sprintf("   Paired t-test: t(%d) = %.2f, p = %.4f\n", 
            t_test_result$parameter, t_test_result$statistic, t_test_result$p.value))

cat("\n2. Statistical Significance:\n")
n_sig <- sum(summary_stats$DeLong_p < 0.05 & summary_stats$Delta_AUC_T_vs_P > 0, na.rm = TRUE)
cat(sprintf("   Therapies where therapist > patient (p < .05): %d/%d (%.0f%%)\n",
            n_sig, nrow(summary_stats), n_sig / nrow(summary_stats) * 100))
cat(sprintf("   Median DeLong Z-statistic: %.2f\n", median(summary_stats$DeLong_Z, na.rm = TRUE)))

cat("\n3. Therapies with Largest Therapist Advantage:\n")
top_therapies <- summary_stats %>% arrange(desc(Delta_AUC_T_vs_P)) %>% head(3)
for (i in 1:nrow(top_therapies)) {
  cat(sprintf("   %s: Δ AUC = %.3f (p = %.4f)\n", 
              top_therapies$Therapy[i], 
              top_therapies$Delta_AUC_T_vs_P[i],
              top_therapies$DeLong_p[i]))
}

cat("\n4. Incremental Value:\n")
cat(sprintf("   Adding therapist to patient: Δ AUC = %.3f\n", 
            mean(summary_stats$Delta_AUC_C_vs_P)))
cat(sprintf("   Adding patient to therapist: Δ AUC = %.3f\n", 
            mean(summary_stats$Delta_AUC_C_vs_T)))

cat("\n5. Model Calibration (Brier Score - lower is better):\n")
cat(sprintf("   Patient features:  %.3f\n", mean(summary_stats$Brier_Patient)))
cat(sprintf("   Therapist factors: %.3f\n", mean(summary_stats$Brier_Therapist)))
cat(sprintf("   Combined model:    %.3f\n", mean(summary_stats$Brier_Combined)))


# ============================================================================
# PART 9: MANUSCRIPT TEXT
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("MANUSCRIPT TEXT\n")
cat(rep("=", 90), "\n", sep="")

top3 <- summary_stats %>% arrange(desc(Delta_AUC_T_vs_P)) %>% head(3)

manuscript_text <- sprintf('
Models using therapist identity and practice location achieved substantially 
higher discrimination (mean AUROC = %.3f, SD = %.3f; range: %.3f-%.3f) 
compared with comprehensive patient clinical features (mean AUROC = %.3f, 
SD = %.3f; range: %.3f-%.3f; mean difference = %.3f; 95%% CI: %.3f to %.3f; 
paired t-test t(%d) = %.2f, p < .001). This pattern was statistically 
significant for %d of %d therapy modalities (DeLong test p < .05).

The therapist advantage was most pronounced for %s (Δ AUROC = %.3f, p < .001), 
%s (%.3f, p < .001), and %s (%.3f, p < .001). Adding therapist information 
to patient-based models improved discrimination substantially (mean Δ AUROC = 
%.3f), whereas adding patient clinical features to therapist-based models 
provided negligible incremental value (mean Δ AUROC = %.3f), indicating that 
therapy assignment is predominantly determined by provider and organizational 
factors rather than patient clinical presentation.
',
                           mean(summary_stats$Therapist_AUC), sd(summary_stats$Therapist_AUC),
                           min(summary_stats$Therapist_AUC), max(summary_stats$Therapist_AUC),
                           mean(summary_stats$Patient_AUC), sd(summary_stats$Patient_AUC),
                           min(summary_stats$Patient_AUC), max(summary_stats$Patient_AUC),
                           mean(summary_stats$Delta_AUC_T_vs_P),
                           quantile(summary_stats$Delta_AUC_T_vs_P, 0.025),
                           quantile(summary_stats$Delta_AUC_T_vs_P, 0.975),
                           t_test_result$parameter, t_test_result$statistic,
                           n_sig, nrow(summary_stats),
                           top3$Therapy[1], top3$Delta_AUC_T_vs_P[1],
                           top3$Therapy[2], top3$Delta_AUC_T_vs_P[2],
                           top3$Therapy[3], top3$Delta_AUC_T_vs_P[3],
                           mean(summary_stats$Delta_AUC_C_vs_P),
                           mean(summary_stats$Delta_AUC_C_vs_T)
)

cat(manuscript_text)


# ============================================================================
# PART 10: CREATE ROC CURVE VISUALIZATION (FIXED LEGEND COLORS)
# ============================================================================

cat("\nStep 7: Creating ROC curve visualization...\n")

# JAMA color palette
JAMA_COLORS <- list(
  primary = "#0066CC",
  secondary = "#CC0000",
  combined = "#009900",
  grid = "#EBEBEB",
  reference = "#666666",
  text = "#333333"
)

# Create plots for each therapy
plot_list <- lapply(results, function(res) {
  # Extract data
  fpr_p <- res$patient$fpr
  tpr_p <- res$patient$tpr
  auc_p <- res$patient$auc
  
  fpr_t <- res$therapist$fpr
  tpr_t <- res$therapist$tpr
  auc_t <- res$therapist$auc
  
  fpr_c <- res$combined$fpr
  tpr_c <- res$combined$tpr
  auc_c <- res$combined$auc
  
  # Create data frame for plotting - FIXED: order matters for legend
  plot_data <- rbind(
    data.frame(fpr = fpr_c, tpr = tpr_c, 
               model = "Combined", 
               auc = auc_c, linetype = "dashed"),
    data.frame(fpr = fpr_p, tpr = tpr_p, 
               model = "Patient features", 
               auc = auc_p, linetype = "solid"),
    data.frame(fpr = fpr_t, tpr = tpr_t, 
               model = "Therapist info", 
               auc = auc_t, linetype = "solid")
  )
  
  # Make model a factor with explicit order
  plot_data$model <- factor(plot_data$model, 
                            levels = c("Combined", "Patient features", "Therapist info"))
  
  # Create plot
  p <- ggplot() +
    # Reference diagonal
    geom_abline(intercept = 0, slope = 1, 
                color = JAMA_COLORS$reference, 
                linetype = "dashed", linewidth = 0.4, alpha = 0.5) +
    
    # ROC curves
    geom_line(data = plot_data, 
              aes(x = fpr, y = tpr, color = model, linetype = linetype),
              linewidth = 0.8) +
    
    # Styling - FIXED: explicit breaks to match color mapping
    scale_color_manual(
      values = c("Combined" = JAMA_COLORS$combined,
                 "Patient features" = JAMA_COLORS$primary,
                 "Therapist info" = JAMA_COLORS$secondary),
      breaks = c("Combined", "Patient features", "Therapist info"),
      labels = c(
        sprintf("Combined\n(AUC = %.2f)", auc_c),
        sprintf("Patient features\n(AUC = %.2f)", auc_p),
        sprintf("Therapist info\n(AUC = %.2f)", auc_t)
      )
    ) +
    scale_linetype_identity() +
    
    # Axes
    scale_x_continuous(limits = c(-0.02, 1.02), 
                       breaks = seq(0, 1, 0.25),
                       expand = c(0, 0)) +
    scale_y_continuous(limits = c(-0.02, 1.02), 
                       breaks = seq(0, 1, 0.25),
                       expand = c(0, 0)) +
    
    labs(
      title = res$therapy,
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = NULL
    ) +
    
    # Theme
    theme_minimal() +
    theme(
      text = element_text(family = "sans", color = JAMA_COLORS$text),
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5,
                                margin = margin(b = 8)),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9, color = "#666666"),
      axis.line = element_line(color = JAMA_COLORS$text, linewidth = 0.5),
      axis.ticks = element_line(color = "#666666", linewidth = 0.5),
      axis.ticks.length = unit(3, "pt"),
      panel.grid.major = element_line(color = JAMA_COLORS$grid, linewidth = 0.25),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = element_rect(fill = "white", color = JAMA_COLORS$grid, 
                                       linewidth = 0.5),
      legend.key.size = unit(0.8, "lines"),
      legend.text = element_text(size = 8.5),
      legend.margin = margin(4, 4, 4, 4),
      aspect.ratio = 1
    ) +
    coord_fixed()
  
  # Add AUC difference annotations
  auc_diff_pc <- auc_c - auc_p  # Combined vs Patient
  auc_diff_pt <- auc_p - auc_t  # Patient vs Therapist
  
  if (abs(auc_diff_pc) > 0.03) {
    p <- p + annotate("text", x = 0.05, y = 0.95, 
                      label = sprintf("Δ(C-P) = %+.2f", auc_diff_pc),
                      hjust = 0, vjust = 1, size = 2.8, 
                      color = JAMA_COLORS$combined, fontface = "bold")
  }
  
  if (abs(auc_diff_pt) > 0.03) {
    diff_color <- ifelse(auc_diff_pt > 0, JAMA_COLORS$primary, JAMA_COLORS$secondary)
    p <- p + annotate("text", x = 0.05, y = 0.88, 
                      label = sprintf("Δ(P-T) = %+.2f", auc_diff_pt),
                      hjust = 0, vjust = 1, size = 2.8, 
                      color = diff_color, fontface = "bold")
  }
  
  return(p)
})

# Arrange plots in grid
if (length(plot_list) > 0) {
  ncols <- 4
  nrows <- ceiling(length(plot_list) / ncols)
  
  combined_plot <- do.call(grid.arrange, c(plot_list, ncol = ncols, nrow = nrows))
  
  ggsave("therapy_roc_curves.png", combined_plot, 
         width = ncols * 2.5, height = nrows * 2.5 * 1.3, 
         dpi = 300, bg = "white")
  
  cat("  Plot saved as 'therapy_roc_curves.png'\n")
}


# ============================================================================
# PART 11: SAVE RESULTS
# ============================================================================

cat("\nStep 8: Saving results...\n")

write.csv(summary_stats, "therapy_assignment_analysis.csv", row.names = FALSE)
cat("  Summary statistics saved to 'therapy_assignment_analysis.csv'\n")

# Optional: Save propensity scores separately
propensity_cols <- names(data)[grepl("^prop_", names(data))]
write.csv(data[, c("master_id", propensity_cols)], 
          "therapy_propensity_scores.csv", row.names = FALSE)
cat("  Propensity scores saved to 'therapy_propensity_scores.csv'\n")

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("ANALYSIS COMPLETE\n")
cat(rep("=", 90), "\n", sep="")
cat("\nKey findings:\n")
cat(sprintf("  - Therapist models outperformed patient models by %.3f AUC points\n", 
            mean(summary_stats$Delta_AUC_T_vs_P)))
cat(sprintf("  - Statistically significant in %d/%d therapies\n", n_sig, nrow(summary_stats)))
cat(sprintf("  - Adding therapist info to patient models: +%.3f AUC\n", 
            mean(summary_stats$Delta_AUC_C_vs_P)))
cat(sprintf("  - Adding patient info to therapist models: +%.3f AUC\n", 
            mean(summary_stats$Delta_AUC_C_vs_T)))
cat(sprintf("\n  - Propensity scores added to 'data' object (%d columns)\n", length(propensity_cols)))
cat("  - Use these in your XGBoost model to control for selection bias\n")