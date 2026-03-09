from datetime import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import *
from feature_engine.imputation import (
    MeanMedianImputer,
    ArbitraryNumberImputer,
    CategoricalImputer,
)
from collections import Counter
from feature_engine.encoding import OneHotEncoder
from feature_engine.outliers import Winsorizer

def assess_data(raw_data):
# Basic data exploration
    print(raw_data.head())
    print(raw_data.info())
    print(raw_data.describe())

# Check for duplicates and missing values
    duplicate_rows = raw_data.duplicated()
    rows_with_missing = raw_data.isnull().any(axis=1)
    columns_with_missing = raw_data.isnull().any(axis=0)
    print("Number of duplicate rows:", duplicate_rows.sum())
    print("Number of rows with missing values:", rows_with_missing.sum())
    print("Number of columns with missing values:", columns_with_missing.sum())

# Categorize features by type
    categorical_features = raw_data.select_dtypes(include=['object']).columns.tolist()
    numerical_features = (raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist())
    binary_features = raw_data.select_dtypes(include=['bool']).columns.tolist()
    return categorical_features, numerical_features, binary_features

def assess_numeric_feature(raw_data, column_name):
    if column_name not in raw_data.columns:
        raise ValueError(f"Feature '{column_name}' not found in the dataset")
    column_data = raw_data[column_name].copy()
    if not pd.api.types.is_numeric_dtype(column_data):
        raise TypeError(f"Feature '{column_data}' is not numeric")

# Calculate statistics
    stats = {
        'count': column_data.count(),
        'missing': column_data.isna().sum(),
        'missing_pct': column_data.isna().mean() * 100,
        'mean': column_data.mean(),
        'median': column_data.median(),
        'std': column_data.std(),
        'min': column_data.min(),
        'max': column_data.max(),
        'q1': column_data.quantile(0.25),
        'q3': column_data.quantile(0.75),
        'iqr': column_data.quantile(0.75) - column_data.quantile(0.25),
        'skewness': column_data.skew()
    }

# Outlier detection using IQR method
    lower_bound = stats['q1'] - 1.5 * stats['iqr']
    upper_bound = stats['q3'] + 1.5 * stats['iqr']
    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
    stats['outliers_count'] = len(outliers)

# Data visualization
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[2, 1])

# Distribution plot with mean and median
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(column_data, kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column_name}')
    ax1.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.2f}")
    ax1.legend()

# Box plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x=column_data, ax=ax2)
    ax2.set_title(f'Box Plot of {column_name}')

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

# Summary statistics
    stats_text = f"""---- {column_name} Statistics ----
    Count: {stats['count']} (Missing: {stats['missing']} - {stats['missing_pct']:.2f}%)
    Central Tendency: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}
    Dispersion: Min = {stats['min']:.2f}, Max = {stats['max']:.2f}, Std = {stats['std']:.2f}
    Quartiles: Q1 = {stats['q1']:.2f}, Q3 = {stats['q3']:.2f}, IQR = {stats['iqr']:.2f}
    Shape: Skewness = {stats['skewness']:.2f}
    Outliers: {stats['outliers_count']} ({stats['outliers_count'] / stats['count'] * 100:.2f}%)"""

    ax3.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.5),
             family='monospace')

    plt.tight_layout()
    plt.show()

def assess_categorical_feature(raw_data, column_name, max_categories=15):
# Basic validation
    if column_name not in raw_data.columns:
        raise ValueError(f"Feature '{column_name}' not found in the dataset")

    column_data = raw_data[column_name].copy()

# Calculate statistics for categorical data
    stats = {
        'count': column_data.count(),
        'missing': column_data.isna().sum(),
        'missing_pct': column_data.isna().mean() * 100,
        'unique_values': column_data.nunique(),
        'mode': column_data.mode()[0] if not column_data.mode().empty else None,
        'mode_count': column_data.value_counts().iloc[0] if not column_data.value_counts().empty else 0,
        'mode_pct': (column_data.value_counts().iloc[0] / column_data.count() * 100) if not column_data.value_counts().empty else 0
    }

    # Sort value counts from highest to lowest
    value_counts = column_data.value_counts().sort_values(ascending=False)
    value_counts_pct = column_data.value_counts(normalize=True).sort_values(ascending=False) * 100

    too_many_categories = len(value_counts) > max_categories
    if too_many_categories:
        top_categories = value_counts.iloc[:max_categories - 1]
        other_count = value_counts.iloc[max_categories - 1:].sum()
        top_categories_pct = value_counts_pct.iloc[:max_categories - 1]
        other_pct = value_counts_pct.iloc[max_categories - 1:].sum()
        value_counts = pd.concat([top_categories, pd.Series([other_count], index=['Other'])])
        value_counts_pct = pd.concat([top_categories_pct, pd.Series([other_pct], index=['Other'])])

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, height_ratios=[2, 2, 1])

    # Bar chart
    ax1 = fig.add_subplot(gs[0, :])
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax1, order=value_counts.index)
    ax1.set_title(f'Count of {column_name} Categories')
    ax1.set_ylabel('Count')
    ax1.set_xlabel(column_name)

    if len(value_counts) > 5:
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    for i, v in enumerate(value_counts.values):
        ax1.text(i, v + (v * 0.01), f'{int(v)}', ha='center')

    # Pie chart
    ax2 = fig.add_subplot(gs[1, 0])
    wedges, texts, autotexts = ax2.pie(
        value_counts.values,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.5)
    )
    ax2.set_title(f'Percentage of {column_name} Categories')

    ax2.legend(
        wedges,
        value_counts.index,
        title=column_name,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # Horizontal bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    sns.barplot(x=value_counts_pct.values, y=value_counts_pct.index, ax=ax3, orient='h', order=value_counts_pct.index)
    ax3.set_title(f'Percentage of {column_name} Categories')
    ax3.set_xlabel('Percentage (%)')
    ax3.set_ylabel(column_name)

    for i, v in enumerate(value_counts_pct.values):
        ax3.text(v + 0.5, i, f'{v:.1f}%', va='center')

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Summary statistics textbox
    stats_text = f"""---- {column_name} Statistics ----
Count: {stats['count']} (Missing: {stats['missing']} - {stats['missing_pct']:.2f}%)
Unique Values: {stats['unique_values']}
Mode (Most Common): {stats['mode']} (Count: {stats['mode_count']}, {stats['mode_pct']:.2f}%)
"""
    if too_many_categories:
        stats_text += f"\nNote: Showing top {max_categories - 1} categories. {value_counts.shape[0] - (max_categories - 1)} less frequent categories grouped as 'Other'."

    ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='mintcream', alpha=0.5),
             family='monospace')

    plt.tight_layout()
    plt.show()

    print(stats_text)

    # Show top categories
    print("\nTop Categories:")
    top_n = min(10, len(column_data.value_counts()))
    for i, (cat, count) in enumerate(column_data.value_counts().iloc[:top_n].items()):
        pct = count / stats['count'] * 100
        print(f"{i + 1}. {cat}: {count} ({pct:.2f}%)")

def convert_feature_type(raw_data, column_name, target_type):
    transformed_data = raw_data.copy()
    if column_name not in transformed_data.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    try:
        if target_type.lower() == 'str' or target_type.lower() == 'string':
            transformed_data[column_name] = transformed_data[column_name].astype(str)
        elif target_type.lower() == 'int' or target_type.lower() == 'integer':
            # Handle NaN values before converting to int
            if transformed_data[column_name].isna().any():
                transformed_data[column_name] = transformed_data[column_name].astype('Int64')
            else:
                transformed_data[column_name] = transformed_data[column_name].astype(int)
        elif target_type.lower() == 'float':
            transformed_data[column_name] = transformed_data[column_name].astype(float)
        elif target_type.lower() == 'bool' or target_type.lower() == 'boolean':
            # Handle string boolean conversion
            if transformed_data[column_name].dtype == 'object':
                true_values = ['true', 'yes', 'y', '1', 't']
                false_values = ['false', 'no', 'n', '0', 'f']
                def map_to_bool(val):
                    if pd.isna(val):
                        return pd.NA
                    if isinstance(val, str):
                        val_lower = val.lower()
                        if val_lower in true_values:
                            return True
                        elif val_lower in false_values:
                            return False
                    return bool(val)
                transformed_data[column_name] = transformed_data[column_name].apply(map_to_bool)
            else:
                transformed_data[column_name] = transformed_data[column_name].astype(bool)
        elif target_type.lower() == 'category':
            transformed_data[column_name] = transformed_data[column_name].astype('category')
        elif target_type.lower() in ('datetime', 'date', 'time'):
            transformed_data[column_name] = pd.to_datetime(transformed_data[column_name], errors='coerce')
        else:
            transformed_data[column_name] = transformed_data[column_name].astype(target_type)
    except Exception as e:
        print(f"Error converting '{column_name}' to {target_type}: {str(e)}")
    return transformed_data

def remove_feature_substr(raw_data, column_name, substr):
    if column_name not in raw_data.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    recoded_data = raw_data.copy()

    if recoded_data[column_name].dtype != 'object':
        print(f"Converting column '{column_name}' to string type before removing substring")
        recoded_data[column_name] = recoded_data[column_name].astype(str)
    original_values = recoded_data[column_name].iloc[:5].tolist()
    recoded_data[column_name] = recoded_data[column_name].str.replace(substr, '', regex=False)
    print(f"Removed '{substr}' from column '{column_name}'")
    print("Examples of transformation:")
    for i, (before, after) in enumerate(zip(original_values, recoded_data[column_name].iloc[:5])):
        print(f"  {i + 1}. '{before}' → '{after}'")
    if not raw_data[column_name].equals(recoded_data[column_name]):
        modified_count = (raw_data[column_name] != recoded_data[column_name]).sum()
        print(f"Modified {modified_count} values out of {len(recoded_data)} ({modified_count / len(recoded_data) * 100:.2f}%)")
    else:
        print("No values were modified. Substring not found in any value.")
    return recoded_data

def impute_missing_numeric_data(raw_data, column_name, method='median', arbitrary_value=0):
    # Basic validation
    if column_name not in raw_data.columns:
        raise ValueError(f"Feature '{column_name}' not found in the dataset")
    
    # Create DataFrame for imputation
    column_data = pd.DataFrame({
        column_name: raw_data[column_name]
    })
    missing_count =   column_data['col'].isna().sum()
    if missing_count == 0:
        print("No missing values in the column")
        return column_data
    
    # Choose imputation method
    if method == 'mean':
        imputer = MeanMedianImputer(imputation_method='mean', variables=['col'])
    elif method == 'median':
        imputer = MeanMedianImputer(imputation_method='median', variables=['col'])
    elif method == 'mode':
        mode_value = column_data.mode()[0]
        imputer = ArbitraryNumberImputer(arbitrary_number=mode_value, variables=['col'])
    elif method == 'arbitrary':
        imputer = ArbitraryNumberImputer(arbitrary_number=arbitrary_value, variables=['col'])
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    # Fit and transform the data
    imputer.fit(column_data)
    imputed_column_data = imputer.transform(column_data)
    print(f"Imputed {missing_count} missing values using {method} method")
    return imputed_column_data['col']

def impute_missing_non_numeric_data(raw_data, column_name, method='frequent', arbitrary_value=None):
    if column_name not in raw_data.columns:
        raise ValueError(f"Feature '{column_name}' not found in the dataset")

    column_data = pd.DataFrame({
        'col': raw_data[column_name]
    })
    missing_count = column_data['col'].isna().sum()
    if missing_count == 0:
        print("No missing values in the column")
        return column_data['col']
    unique_values = column_data['col'].dropna().unique()
    is_binary = len(unique_values) <= 2
    if method == 'frequent':
        imputation_method = 'frequent'
    elif method == 'missing':
        if is_binary and len(unique_values) == 2:
            print("Warning: 'missing' method adds a third category to binary data, which breaks binary nature.")
        imputation_method = 'missing'
    elif method == 'arbitrary' or method == 'constant':
        if arbitrary_value is None:
            raise ValueError("arbitrary_value must be provided when using 'arbitrary' or 'constant' method")
        imputation_method = 'constant'
    else:
        raise ValueError(f"Unknown imputation method: {method}. Use 'frequent', 'missing', or 'constant'")
    imputer = CategoricalImputer(
        imputation_method=imputation_method,
        variables=['col'],
        constant=arbitrary_value if imputation_method == 'constant' else None
    )
    imputer.fit(column_data)
    imputed_column_data = imputer.transform(column_data)
    print(f"Imputed {missing_count} missing values using {method} method")
    return imputed_column_data['col']

def recode_numeric_outliers(raw_data, column_name, method='iqr', tail='both', fold=1.5):
    # Validation checks
    if column_name not in raw_data.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset")
    if not pd.api.types.is_numeric_dtype(raw_data[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    column_data = pd.DataFrame({column_name: raw_data[column_name].copy()})
    if method == 'iqr':
        q1 = column_data[column_name].quantile(0.25)
        q3 = column_data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - fold * iqr
        upper_bound = q3 + fold * iqr
    else:
        mean = column_data[column_name].mean()
        std = column_data[column_name].std()
        lower_bound = mean - fold * std
        upper_bound = mean + fold * std
    lower_outliers = 0
    upper_outliers = 0
    if tail in ['left', 'both']:
        lower_outliers = (column_data[column_name] < lower_bound).sum()
    if tail in ['right', 'both']:
        upper_outliers = (column_data[column_name] > upper_bound).sum()
    total_outliers = lower_outliers + upper_outliers
    if total_outliers == 0:
        print("No outliers detected")
        return column_data[column_name]
    print(f"Detected {total_outliers} outliers ({lower_outliers} lower, {upper_outliers} upper)")
    winsor = Winsorizer(
        capping_method=method,
        tail=tail,
        fold=fold,
        variables=[column_name]
    )
    recoded_column_data = winsor.fit_transform(column_data)
    print(f"Outliers capped using {method} method with fold={fold}")
    return recoded_column_data[column_name]

def drop_column(raw_data, column_name,handling_missing=True,threshold=0.3):
    cleaned_data = raw_data.copy()
    if column_name not in cleaned_data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    if handling_missing:
        missing_proportion = cleaned_data[column_name].isnull().mean()
        missing_percentage = missing_proportion * 100
        if missing_proportion > threshold:
            print(
                f"Dropping column '{column_name}': {missing_percentage:.1f}% missing values (exceeds {threshold * 100:.1f}% threshold)")
            return cleaned_data.drop(columns=[column_name])
        else:
            print(
                f"Keeping column '{column_name}': {missing_percentage:.1f}% missing values (below {threshold * 100:.1f}% threshold)")
            return cleaned_data
    else:
        print(f"Dropping column '{column_name}'")
        return cleaned_data.drop(columns=[column_name])

def drop_rows(raw_data, column_name=None, row_indices=None, handling_missing=True, threshold=0.3):
    cleaned_data=raw_data.copy()
    original_count = len(cleaned_data)
    if not handling_missing:
        if isinstance(row_indices, (int, float)):
            row_indices = [int(row_indices)]
        valid_indices = [idx for idx in row_indices if idx in cleaned_data.index]
        invalid_indices = [idx for idx in row_indices if idx not in cleaned_data.index]
        if invalid_indices or row_indices is None:
            print(f"Warning: Row indices {invalid_indices} not found in DataFrame")
        if valid_indices:
            cleaned_data = cleaned_data.drop(valid_indices)
            dropped_count = len(valid_indices)
            dropped_percentage = (dropped_count / original_count) * 100 if original_count > 0 else 0
            print(f"Dropped {dropped_count} rows ({dropped_percentage:.1f}%) based on specified indices")
        return cleaned_data
    
    # Handle missing values
    if column_name is not None:
        if column_name not in cleaned_data.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame")
        if handling_missing:
            rows_to_keep = cleaned_data[column_name].isnull()
            cleaned_data = cleaned_data[rows_to_keep]
        else:
            cleaned_data = cleaned_data.dropna(subset=[column_name])
    else:
        if handling_missing:
            missing_proportions = cleaned_data.isnull().mean(axis=1)
            rows_to_keep = missing_proportions <= threshold
            cleaned_data = cleaned_data[rows_to_keep]
        else:
            cleaned_data = cleaned_data.dropna()

    dropped_count = original_count - len(cleaned_data)
    dropped_percentage = (dropped_count / original_count) * 100 if original_count > 0 else 0
    if column_name:
        print(f"Dropped {dropped_count} rows ({dropped_percentage:.1f}%) with missing values in column '{column_name}'")
    else:
        if handling_missing:
            print(
                f"Dropped {dropped_count} rows ({dropped_percentage:.1f}%) with more than {threshold * 100:.1f}% missing values across all columns")
        else:
            print(f"Dropped {dropped_count} rows ({dropped_percentage:.1f}%) with any missing values")
    return cleaned_data

def drop_duplicates(raw_data, column_names=None, keep='first', reset_index=False):
    cleaned_data = raw_data.copy()
    original_count = len(cleaned_data)
    cleaned_data = cleaned_data.drop_duplicates(subset=column_names, keep=keep)
    if reset_index:
        cleaned_data = cleaned_data.reset_index(drop=True)
    dropped_count = original_count - len(cleaned_data)
    dropped_percentage = (dropped_count / original_count) * 100 if original_count > 0 else 0
    col_text = "all columns" if column_names is None else f"column(s): {', '.join(column_names)}"
    keep_text = "keeping first" if keep == 'first' else "keeping last" if keep == 'last' else "removing all"
    print(f"Dropped {dropped_count} duplicates ({dropped_percentage:.1f}%) based on {col_text}, {keep_text}")
    return cleaned_data

def encode_dummy(raw_data, column_names, drop_first=False, sparse=False):
    transformed_data = raw_data.copy()
    if not column_names:
        return transformed_data
    
    # Use feature engine's OneHotEncoder
    encoder = OneHotEncoder(
        sparse_output=sparse,
        drop='first' if drop_first else None,
        handle_unknown='ignore'
    )

    # Fit and transform the data
    encoded_array = encoder.fit_transform(transformed_data[column_names])
    feature_names = encoder.get_feature_names_out(column_names)
    if sparse:
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded_array,
            index=transformed_data.index,
            columns=feature_names
        )
    else:
        encoded_df = pd.DataFrame(
            encoded_array,
            index=transformed_data.index,
            columns=feature_names
        )
    transformed_data = pd.concat([transformed_data.drop(columns=column_names), encoded_df], axis=1)
    return transformed_data

def recode_to_categorical(raw_data, column_name, mapping_dict=None):
    recoded_data = raw_data.copy()
    if column_name not in recoded_data.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    unique_values = sorted(recoded_data[column_name].dropna().unique())

    if mapping_dict is None:
        mapping_dict = {}
        print(f"Creating mapping for column '{column_name}'")
        print("Enter category label for each value (press Enter to skip):")

        for value in unique_values:
            label = input(f"Label for {value}: ").strip()
            if label:
                mapping_dict[value] = label

    new_column_name = f"{column_name}_recoded"

    recoded_data[new_column_name] = recoded_data[column_name].map(
        lambda x: mapping_dict.get(x, x)
    )
    recoded_data[new_column_name] = recoded_data[new_column_name].astype('category')
    return recoded_data

def bin_numeric_data(raw_data, column_names, method='equal_width', n_bins=10, labels=None):
    transformed_data = raw_data.copy()
    if method not in ['equal_width', 'equal_freq']:
        raise ValueError("Method must be one of ['equal_width', 'equal_freq']")
    for col in column_names:
        if col not in transformed_data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(transformed_data[col]):
            continue
        if method == 'equal_width':
            transformed_data[f"{col}_binned"] = pd.cut(raw_data[col], bins=n_bins, labels=labels)
        elif method == 'equal_freq':
            try:
                transformed_data[f"{col}_binned"] = pd.qcut(raw_data[col], q=n_bins, labels=labels)
            except ValueError:
                transformed_data[f"{col}_binned"] = pd.cut(raw_data[col], bins=n_bins, labels=labels)
    return transformed_data

def normalize_numeric_data(raw_data, column_names, method='min-max'):
    transformed_data = raw_data.copy()
    valid_methods = ['min-max', 'z-score', 'robust', 'max-abs', 'log']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    for col in column_names:
        if col not in transformed_data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(transformed_data[col]):
            continue
        if transformed_data[col].nunique() <= 1:
            continue
        if method == 'log':
            min_val = transformed_data[col].min()
            offset = 0 if min_val > 0 else abs(min_val) + 1e-6
            transformed_data[col] = np.log(transformed_data[col] + offset)
            continue

        # Select scaler based on method
        if method == 'min-max':
            scaler = MinMaxScaler()
        elif method == 'z-score':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'max-abs':
            scaler = MaxAbsScaler()
        values = transformed_data[col].values.reshape(-1, 1)
        transformed_data[col] = scaler.fit_transform(values).flatten()
    return transformed_data

def convert_list_column_to_binary(raw_data, column_name, threshold=None, top_n=None):
    transformed_data = raw_data.copy()
    transformed_data[column_name] = transformed_data[column_name].fillna('[]')
    if isinstance(transformed_data[column_name].iloc[0], str):
        try:
            import ast
            transformed_data[column_name] = transformed_data[column_name].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        except (ValueError, SyntaxError):
            transformed_data[column_name] = transformed_data[column_name].str.strip('[]')
            transformed_data[column_name] = transformed_data[column_name].str.split(',')
            transformed_data[column_name] = transformed_data[column_name].apply(
                lambda x: [item.strip().strip("'\"") for item in x] if isinstance(x, list) else []
            )
    counter = Counter()
    for item_list in transformed_data[column_name]:
        if isinstance(item_list, list):
            counter.update(item_list)
    if top_n:
        frequent_items = [item for item, count in counter.most_common(top_n)]
    elif threshold:
        frequent_items = [item for item, count in counter.items() if count >= threshold]
    else:
        frequent_items = [item for item, count in counter.items()]
    for value in frequent_items:
        if not value or pd.isna(value):
            continue
        col_name = f"{column_name}_{str(value).replace(' ', '_').lower()}"
        transformed_data[col_name] = transformed_data[column_name].apply(
            lambda x: 1 if isinstance(x, list) and value in x else 0
        )
    transformed_data[column_name] = transformed_data[column_name].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )
    return transformed_data

def extract_datetime_features(raw_data, column_name):
    transformed_data = raw_data.copy()
    transformed_data[column_name] = pd.to_datetime(transformed_data[column_name],errors='coerce')
    sample = transformed_data[column_name].dropna().head(1000)
    has_time_component = False
    if len(sample) > 0:
        times = sample.dt.time
        has_time_component = any(t != time(0, 0, 0) for t in times if t is not None)

    # Define all possible date features
    date_features = {
        'year': lambda x: x.dt.year,
        'quarter': lambda x: x.dt.quarter,
        'month': lambda x: x.dt.month,
        'day': lambda x: x.dt.day,
        'day_of_week': lambda x: x.dt.dayofweek,
        'week_of_year': lambda x: x.dt.isocalendar().week,
        'day_of_year': lambda x: x.dt.dayofyear,
        'is_month_start': lambda x: x.dt.is_month_start.astype(int),
        'is_month_end': lambda x: x.dt.is_month_end.astype(int),
        'is_quarter_start': lambda x: x.dt.is_quarter_start.astype(int),
        'is_quarter_end': lambda x: x.dt.is_quarter_end.astype(int),
        'is_year_start': lambda x: x.dt.is_year_start.astype(int),
        'is_year_end': lambda x: x.dt.is_year_end.astype(int),
        'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int),
        'is_weekday': lambda x: (~x.dt.dayofweek.isin([5, 6])).astype(int),
        'decade': lambda x: (x.dt.year // 10) * 10,
    }

    time_features = {
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'second': lambda x: x.dt.second,
        'is_morning': lambda x: ((x.dt.hour >= 5) & (x.dt.hour < 12)).astype(int),
        'is_afternoon': lambda x: ((x.dt.hour >= 12) & (x.dt.hour < 18)).astype(int),
        'is_night': lambda x: ((x.dt.hour >= 18) | (x.dt.hour < 5)).astype(int),
    }

    available_features = date_features.copy()
    if has_time_component:
        available_features.update(time_features)

    # Extract selected features
    for feature in available_features:
        transformed_data[f"{column_name}_{feature}"] = available_features[feature](transformed_data[column_name])
    return transformed_data