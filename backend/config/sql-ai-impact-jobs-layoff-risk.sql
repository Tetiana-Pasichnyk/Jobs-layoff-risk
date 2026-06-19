-- 1. Wipe out any old versions + Create the Database
DROP DATABASE IF EXISTS AI_Impact_DB;
CREATE DATABASE AI_Impact_DB;
USE AI_Impact_DB;

-- ==========================================
-- 2. Create Lookup Tables (Dimension Tables)
-- ==========================================

CREATE TABLE education_levels (
    education_id INT AUTO_INCREMENT PRIMARY KEY,
    education_level VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE industries (
    industry_id INT AUTO_INCREMENT PRIMARY KEY,
    industry_name VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE job_roles (
    job_role_id INT AUTO_INCREMENT PRIMARY KEY,
    job_role_name VARCHAR(100) NOT NULL UNIQUE
);

-- ==========================================
-- 3. Create Main Entity Tables
-- ==========================================

CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    age DECIMAL(3,1),
    years_of_experience DECIMAL(4,1),
    education_id INT,
    industry_id INT,
    job_role_id INT,
    company_size VARCHAR(20),  -- e.g., Small, Medium, Large
    job_level VARCHAR(20),     -- e.g., Entry, Mid, Senior
    FOREIGN KEY (education_id) REFERENCES education_levels(education_id),
    FOREIGN KEY (industry_id) REFERENCES industries(industry_id),
    FOREIGN KEY (job_role_id) REFERENCES job_roles(job_role_id)
);

-- ==========================================
-- 4. Create Performance & Risk Metrics Table
-- ==========================================

CREATE TABLE ai_impact_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT UNIQUE, -- 1:1 relationship with employee for specific survey instance
    routine_task_percentage DECIMAL(5,2),
    creativity_requirement DECIMAL(5,2),
    human_interaction_level DECIMAL(5,2),
    ai_adoption_level VARCHAR(20), -- Low, Medium, High
    number_of_ai_tools_used DECIMAL(4,1),
    ai_usage_hours_per_week DECIMAL(5,2),
    tasks_automated_percentage DECIMAL(5,2),
    ai_training_hours DECIMAL(5,2),
    layoff_risk VARCHAR(20),        -- Low, Medium, High
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
);