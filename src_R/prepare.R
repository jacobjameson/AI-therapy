library(readxl)
library(janitor)
library(dplyr)
library(stringr)
library(forcats)
library(tidyr)
library(lubridate)
library(fuzzyjoin)

############################################################################
mh_sa <- read_excel("~/Sue Goldie Dropbox/Jacob Jameson/DBH data/MH SA.xlsx")
mh_sa <- clean_names(mh_sa)

mh_sa <- mh_sa %>%
  rename(
    risk_final = final_b_risk_level_b_including_any_change_based_on_clinical_judgment_if_applicable_u,
    risk_high_click = please_click_here_for_high_risk_level_if_applicable,
    risk_mod_click = please_click_here_for_moderate_risk_level_if_applicable,
    risk_low_click = please_click_here_for_low_risk_level_if_applicable
  )

mh_sa <- mh_sa %>%
  rowwise() %>%
  mutate(
    risk_level_initial = case_when(
      !is.na(risk_low_click) && str_trim(tolower(as.character(risk_low_click))) %in% c("", "na", "n/a") == FALSE ~ "Low",
      !is.na(risk_mod_click) && str_trim(tolower(as.character(risk_mod_click))) %in% c("", "na", "n/a") == FALSE ~ "Moderate",
      !is.na(risk_high_click) && str_trim(tolower(as.character(risk_high_click))) %in% c("", "na", "n/a") == FALSE ~ "High",
      !is.na(risk_level) & str_detect(tolower(risk_level), "low") ~ "Low",
      !is.na(risk_level) & str_detect(tolower(risk_level), "moderate") ~ "Moderate",
      !is.na(risk_level) & str_detect(tolower(risk_level), "high") ~ "High",
      TRUE ~ NA_character_
    )
  ) %>%
  ungroup() %>%
  mutate(risk_level_initial = factor(risk_level_initial, 
                                     levels = c("Low", "Moderate", "High"),
                                     ordered = TRUE))

# Display counts
table(mh_sa$risk_level_initial, useNA = "ifany")

############################################################################
# --- Read and clean ---
srs <- read_excel("~/Sue Goldie Dropbox/Jacob Jameson/DBH data/MH SRS.xlsx") %>%
  janitor::clean_names() %>%
  rename(
    raw_risk_level       = final_b_risk_level_b_including_any_change_based_on_clinical_judgment_if_applicable_u,
    high_risk_flag1      = documentation_follow_up_create_resolve_urgent_issue_red_flag_in_emr_high_risk,
    high_risk_flag2      = documentation_follow_up_create_suicide_risk_treatment_plan_high_risk,
    moderate_risk_flag   = documentation_follow_up_include_suicide_risk_reduction_interventions_in_appropriate_treatment_plan_moderate_risk,
    low_risk_flag        = documentation_follow_up_n_a_only_applicable_if_low_risk
  )

# --- Normalize flag columns inline ---
srs <- srs %>%
  mutate(
    high_risk_flag1    = tolower(str_trim(as.character(high_risk_flag1))) %in% c("true", "t", "yes", "y", "1"),
    high_risk_flag2    = tolower(str_trim(as.character(high_risk_flag2))) %in% c("true", "t", "yes", "y", "1"),
    moderate_risk_flag = tolower(str_trim(as.character(moderate_risk_flag))) %in% c("true", "t", "yes", "y", "1"),
    low_risk_flag      = tolower(str_trim(as.character(low_risk_flag))) %in% c("true", "t", "yes", "y", "1")
  )

# --- Map raw_risk_level to standardized categories inline ---
srs <- srs %>%
  mutate(
    initial_risk_level = case_when(
      str_to_lower(str_trim(raw_risk_level)) == "low suicide risk"      ~ "Low",
      str_to_lower(str_trim(raw_risk_level)) == "moderate suicide risk" ~ "Moderate",
      str_to_lower(str_trim(raw_risk_level)) == "high suicide risk"     ~ "High",
      TRUE ~ NA_character_
    )
  )

# --- Priority-based flag risk ---
srs <- srs %>%
  mutate(
    flag_based_risk = case_when(
      high_risk_flag1 | high_risk_flag2 ~ "High",
      moderate_risk_flag                ~ "Moderate",
      low_risk_flag                     ~ "Low",
      TRUE                              ~ NA_character_
    )
  )

# --- Coalesce: use raw field first, then flag-based ---
srs <- srs %>%
  mutate(
    risk_level = coalesce(initial_risk_level, flag_based_risk),
    risk_level = factor(risk_level, levels = c("Low", "Moderate", "High"), ordered = TRUE)
  )

# --- Drop intermediate columns and sort ---
srs <- srs %>%
  select(-raw_risk_level, -high_risk_flag1, -high_risk_flag2, -moderate_risk_flag,
         -low_risk_flag, -initial_risk_level, -flag_based_risk) %>%
  arrange(master_id, admission_date, evaluation_date)

# --- Sanity check ---
print(table(srs$risk_level, useNA = "ifany"))


# keep only the first and last SRS for each admission
srs <- srs %>%
  group_by(master_id, admission_date) %>%
  filter(evaluation_date == min(evaluation_date) | evaluation_date == max(evaluation_date)) %>%
  ungroup()

# create a new variable to indicate if the SRS is the first or last for the admission that allows for it to be both
srs <- srs %>%
  group_by(master_id, admission_date) %>%
  mutate(srs_first = ifelse(evaluation_date == min(evaluation_date), 1, 0),
         srs_last = ifelse(evaluation_date == max(evaluation_date), 1, 0)) %>%
  ungroup()

srs <- srs %>%
  select(master_id, admission_date, evaluation_date, risk_level, srs_first, srs_last)


srs <- srs %>%
  group_by(master_id, admission_date) %>%
  reframe(
    risk_level_srs_first = risk_level[srs_first == 1][1],
    risk_level_srs_last = risk_level[srs_last == 1][1],
    days_first_srs = as.numeric(difftime(evaluation_date[srs_first == 1][1], admission_date, units = "days")),
    days_last_srs = as.numeric(difftime(evaluation_date[srs_last == 1][1], admission_date, units = "days")),
    .groups = "drop"
  ) %>%
  unique()

############################################################################
pn <- read_excel("~/Sue Goldie Dropbox/Jacob Jameson/DBH data/MH PN.xlsx")
pn <- clean_names(pn)

pn <- pn %>%
  rename(
    pn_evaluation_date = evaluation_date,
    therapist = staff_signature_1,
    act = evidence_based_modalities_employed_act,
    cbt = evidence_based_modalities_employed_cbt,
    dbt = evidence_based_modalities_employed_dbt,
    motivational_interviewing = evidence_based_modalities_employed_motivational_interviewing,
    mindfulness = evidence_based_modalities_employed_mindfulness_techniques,
    stages_of_change = evidence_based_modalities_employed_stages_of_change,
    family_systems = evidence_based_modalities_employed_family_systems,
    trauma_informed = evidence_based_modalities_employed_trauma_informed_strategies_inc_emdr
  ) %>%
  select(therapist, pn_evaluation_date, master_id, act:trauma_informed, session_type, time_service_started_ended)

ids <- unique(mh_sa$master_id)

pn <- pn %>%
  filter(master_id %in% ids)

pn <- merge(select(mh_sa, master_id, admission_date, discharge_date), 
                pn, by = "master_id", all.x = TRUE, suffixes = c("_mhsa", "_pn"))

# keep only PN evaluations within MH SA admission-discharge window
pn <- pn %>%
  filter(pn_evaluation_date >= admission_date & pn_evaluation_date <= discharge_date)

# keep first PN evaluation per MH SA admission
pn <- pn %>%
  arrange(master_id, admission_date, pn_evaluation_date) %>%
  group_by(master_id, admission_date, discharge_date) %>%
  slice_head(n = 1) %>%
  ungroup()

get_duration_minutes <- function(x) {
  # Handle numeric values (e.g., Excel serials)
  if (suppressWarnings(!is.na(as.numeric(x)))) return(NA_real_)
  
  # Split start and end
  parts <- strsplit(x, "/")[[1]]
  if (length(parts) != 2) return(NA_real_)
  
  # Parse using lubridate
  start_time <- parse_date_time(parts[1], orders = "ymd HMS z", tz = "UTC")
  end_time   <- parse_date_time(parts[2], orders = "ymd HMS z", tz = "UTC")
  
  # Return duration in minutes
  if (is.na(start_time) | is.na(end_time)) return(NA_real_)
  return(as.numeric(difftime(end_time, start_time, units = "mins")))
}

pn$therapy_duration_minutes <- sapply(pn$time_service_started_ended, get_duration_minutes)

table(pn$therapy_duration_minutes)

pn <- pn %>%
  mutate(
    therapy_duration_category = case_when(
      therapy_duration_minutes < 30  ~ "00–30m",
      therapy_duration_minutes >= 30 & therapy_duration_minutes < 45  ~ "30–45m",
      therapy_duration_minutes >= 45 & therapy_duration_minutes <= 60 ~ "45–60m",
      therapy_duration_minutes > 60  ~ "60m+",
      TRUE                                              ~ NA_character_
    ),
    therapy_duration_category = factor(
      therapy_duration_category,
      levels = c("00–30m", "30–45m", "45–60m", "60m+"),
      ordered = TRUE
    ))
  
pn <- pn %>%
  mutate(
    delivery_method = case_when(
      str_detect(session_type, "^Telehealth") ~ "Telehealth",
      TRUE                                    ~ "In-person"
    ),
    session_mode = case_when(
      str_detect(session_type, "Individual")        ~ "Individual",
      str_detect(session_type, "Family")            ~ "Family",
      str_detect(session_type, "Collateral Contact")~ "Collateral Contact",
      TRUE                                          ~ NA_character_
    ),
    delivery_method = factor(delivery_method, levels = c("In-person", "Telehealth")),
    session_mode = factor(session_mode, levels = c("Individual", "Family", "Collateral Contact"))
  ) %>% select(-session_type, -time_service_started_ended, -therapy_duration_minutes)

###########################################################################

# merge SRS and PN data with mh_sa
mh_sa <- merge(mh_sa, srs, by = c("master_id", "admission_date"), all.x = TRUE)
mh_sa <- merge(mh_sa, pn, by = c("master_id", "admission_date", "discharge_date"), all.x = TRUE)

# drop if risk_level_srs_first is NA (i.e., no SRS data)
mh_sa <- mh_sa %>%
  filter(!is.na(risk_level_srs_first))

mh_sa <- mh_sa %>%
  mutate(intake_to_pn = as.numeric(difftime(pn_evaluation_date, evaluation_date, units = "days")))
mh_sa$intake_to_pn[mh_sa$intake_to_pn < 0] <- 0


mh_sa <- mh_sa[,c(1:88, 155:159,161:174)]

mh_sa <- mh_sa %>%
  mutate(across(where(is.logical), as.integer))

mh_sa <- mh_sa %>%
  mutate(across(
    c(
      "how_many_times_have_you_had_these_thoughts",
      "when_you_have_the_thoughts_how_long_do_they_last",
      "frequency_month",
      "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to",
      "are_there_things",
      "duration_month",
      "what_sort_of_reasons",
      "controllability_month",
      "deterrents_month"
    ),
    ~ as.numeric(str_extract(.x, "\\d+"))
  ))

mh_sa <- mh_sa %>%
  mutate(across(
    c(
      "how_many_times_have_you_had_these_thoughts",
      "when_you_have_the_thoughts_how_long_do_they_last",
      "frequency_month",
      "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to",
      "are_there_things",
      "duration_month",
      "what_sort_of_reasons",
      "controllability_month",
      "deterrents_month"
    ),
    ~ replace_na(.x, 0)
  ))

# for the therapy variables fill with 0 if NA
mh_sa <- mh_sa %>%
  mutate(across(
    c(
      "act", "cbt", "dbt", "mindfulness", 
      "motivational_interviewing", "stages_of_change", 
      "family_systems", "trauma_informed"
    ),
    ~ replace_na(.x, 0)
  ))


# Identify columns with more than 5% missing
cols_to_drop <- names(which(colMeans(is.na(mh_sa)) > 0.05))
mh_sa <- mh_sa[, !(names(mh_sa) %in% cols_to_drop)]


mh_sa <- mh_sa %>%
  filter(!is.na(risk_level_initial), !is.na(risk_level_srs_first)) 

# Therapist name: everything before first comma
mh_sa$therapist_name <- sub(",.*", "", mh_sa$therapist)

# drop rows where the therapist name is missing
mh_sa <- mh_sa %>%
  filter(!is.na(therapist_name) & therapist_name != "")

mh_sa <- mh_sa %>%
  mutate(
    access_to_lethal_methods_does_patient_have_access_to_means_including_firearms_in_the_home = case_when(
      access_to_lethal_methods_does_patient_have_access_to_means_including_firearms_in_the_home == "Yes" ~ 1,
      access_to_lethal_methods_does_patient_have_access_to_means_including_firearms_in_the_home == "No"  ~ 0,
      TRUE                                                                                             ~ NA_real_
    )
  )

mh_sa <- mh_sa %>%
  mutate(across(6:66, ~ replace_na(.x, 0)))

mh_sa <- mh_sa %>%
  filter((is.na(days_last_srs) | days_last_srs >= 0) & 
           (is.na(days_first_srs) | days_first_srs >= 0))

mh_sa <- mh_sa %>%
  mutate(
    # Year and month (e.g. "2023-09")
    pn_year = format(pn_evaluation_date, "%Y"),
    pn_month = format(pn_evaluation_date, "%m"),
    
    # Hour of day (0 to 23)
    pn_hour = hour(pn_evaluation_date),
    
    # 4-hour time block (e.g. "00:00-03:59", "04:00-07:59", ...)
    pn_time_block = case_when(
      pn_hour < 4 ~ "00:00–03:59",
      pn_hour < 8 ~ "04:00–07:59",
      pn_hour < 12 ~ "08:00–11:59",
      pn_hour < 16 ~ "12:00–15:59",
      pn_hour < 20 ~ "16:00–19:59",
      TRUE ~ "20:00–23:59"
    )
  ) %>%
  select(-pn_evaluation_date)

mh_sa <- mh_sa %>%
  select(-date, -evaluation_date) %>%
  mutate(session_mode = ifelse(is.na(session_mode), "Individual", session_mode)) 

#############################################################################

demo <- read_excel("~/Sue Goldie Dropbox/Jacob Jameson/DBH data/demo.xlsx")
demo <- clean_names(demo) %>%
  filter(division == 'MH') %>%
  select(master_id, admission_date, location, program, age_group, prim_mh_dx, sex_fs)

mh_sa <- merge(mh_sa, demo, by = c("master_id", "admission_date"), all.x = TRUE)

# Clean and recode primary mental health diagnoses
library(stringr)
library(dplyr)

mh_sa <- mh_sa %>%
  mutate(
    dx_group = case_when(
      # Depressive disorders
      str_detect(prim_mh_dx, regex("depress|mood disorder", ignore_case = TRUE)) ~ "Depressive Disorder",
      
      # Bipolar disorders
      str_detect(prim_mh_dx, regex("bipolar|cyclothymic", ignore_case = TRUE)) ~ "Bipolar Disorder",
      
      # Anxiety disorders (GAD, panic, social anxiety, phobias, etc.)
      str_detect(prim_mh_dx, regex("anxiety|panic|phobia", ignore_case = TRUE)) ~ "Anxiety Disorder",
      
      # Trauma and stressor-related disorders (PTSD, RAD, etc.)
      str_detect(prim_mh_dx, regex("trauma|stress|ptsd|reactive attachment", ignore_case = TRUE)) ~ "Trauma-Related Disorder",
      
      # Substance use
      str_detect(prim_mh_dx, regex("use disorder|substance|alcohol|cannabis|opioid|cocaine|amphetamine|hallucinogen|tobacco", ignore_case = TRUE)) ~ "Substance Use Disorder",
      
      # Neurodevelopmental
      str_detect(prim_mh_dx, regex("adhd|autism|neurodevelopmental|intellectual", ignore_case = TRUE)) ~ "Neurodevelopmental Disorder",
      
      # Personality
      str_detect(prim_mh_dx, regex("personality", ignore_case = TRUE)) ~ "Personality Disorder",
      
      # Psychotic
      str_detect(prim_mh_dx, regex("schizo|psychotic|delusional", ignore_case = TRUE)) ~ "Psychotic Disorder",
      
      # Eating
      str_detect(prim_mh_dx, regex("anorexia|bulimia|eating|feeding", ignore_case = TRUE)) ~ "Eating Disorder",
      
      # Disruptive/impulse control
      str_detect(prim_mh_dx, regex("disruptive|conduct|oppositional|impulse|explosive", ignore_case = TRUE)) ~ "Disruptive/Impulse Disorder",
      
      # OCD and related
      str_detect(prim_mh_dx, regex("obsessive|ocd|trichotillomania", ignore_case = TRUE)) ~ "OCD and Related",
      
      # Sleep/wake disorders
      str_detect(prim_mh_dx, regex("insomnia|sleep", ignore_case = TRUE)) ~ "Sleep Disorder",
      
      # Somatic symptom-related
      str_detect(prim_mh_dx, regex("somatic", ignore_case = TRUE)) ~ "Somatic Symptom Disorder",
      
      # Relational or V-code
      str_detect(prim_mh_dx, regex("relational|parent child|relationship", ignore_case = TRUE)) ~ "Relational/Other V-code",
      
      # No diagnosis
      str_detect(prim_mh_dx, regex("no dx", ignore_case = TRUE)) ~ "No Diagnosis",
      
      # Catch-all
      TRUE ~ "Other/Unspecified"
    )
  ) %>%
  select(-prim_mh_dx)


# if program or location is missing drop
mh_sa <- mh_sa %>%
  filter(!is.na(program) & program != "", !is.na(location) & location != "")


data <- mh_sa %>%
  filter(risk_level_initial != "Low")

