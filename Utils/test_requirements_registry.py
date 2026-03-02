TEST_REQUIREMENTS: dict[str, dict] = {
    "Simple Linear Regression": {
        "dependent": "continuous", "independent": "continuous",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 20,
    },
    "Multiple Linear Regression": {
        "dependent": "continuous", "independent": "any",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 30,
    },
    "Ridge Regression": {
        "dependent": "continuous", "independent": "any",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 30,
    },
    "Lasso Regression": {
        "dependent": "continuous", "independent": "any",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 30,
    },
    "One-Way ANOVA": {
        "dependent": "continuous", "independent": None,
        "grouping": True, "min_groups": 3, "max_groups": None, "min_n": 20,
    },
    "Two-Way ANOVA": {
        "dependent": "continuous", "independent": None,
        "grouping": True, "min_groups": 2, "max_groups": None, "min_n": 30,
    },
    "Independent Samples T-Test": {
        "dependent": "continuous", "independent": None,
        "grouping": True, "min_groups": 2, "max_groups": 2, "min_n": 10,
    },
    "Paired T-Test": {
        "dependent": "continuous", "independent": "continuous",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 10,
    },
    "One Sample T-Test": {
        "dependent": "continuous", "independent": None,
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 5,
    },
    "Mann-Whitney U Test": {
        "dependent": "continuous", "independent": None,
        "grouping": True, "min_groups": 2, "max_groups": 2, "min_n": 10,
    },
    "Wilcoxon Signed-Rank Test": {
        "dependent": "continuous", "independent": "continuous",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 10,
    },
    "Kruskal-Wallis Test": {
        "dependent": "continuous", "independent": None,
        "grouping": True, "min_groups": 3, "max_groups": None, "min_n": 15,
    },
    "Principal Component Analysis": {
        "dependent": None, "independent": "continuous",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 50,
    },
    "Pearson Correlation": {
        "dependent": "continuous", "independent": "continuous",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 10,
    },
    "Spearman Correlation": {
        "dependent": "any", "independent": "any",
        "grouping": False, "min_groups": None, "max_groups": None, "min_n": 10,
    },
}

