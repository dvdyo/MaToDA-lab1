import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    return mo, pd, plt, sns


@app.cell
def _(pd):
    file_path = 'datasets/healthcare-dataset-stroke-data.csv'
    df = pd.read_csv(file_path)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    # Cleaning
    df.dropna(subset=['bmi'], inplace = True)
    df.info()
    return


@app.cell
def _(df, plt, sns):
    # Create a box plot to visualize the distribution and find outliers
    # Box Plot for BMI (before outlier removal)")
    sns.boxplot(x=df['bmi'])
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    As we can see from the here, upper bound is ~47.
    However, to be completely sure, let's calculate this stuff
    """
    )
    return


@app.cell
def _(df):
    # The bounds calculation
    Q1 = df['bmi'].quantile(0.25)
    Q3 = df['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find the indices of the rows that are considered outliers
    outlier_indices = df[(df['bmi'] < lower_bound) | (df['bmi'] > upper_bound)].index

    print(f"Found and removing **{len(outlier_indices)}** outlier rows.")

    df.drop(outlier_indices, inplace=True)
    return lower_bound, upper_bound


@app.cell
def _(lower_bound):
    lower_bound
    return


@app.cell
def _(upper_bound):
    upper_bound
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df, plt, sns):
    # Box Plot for BMI (Before after outlier and NaN removal)
    sns.boxplot(x=df['bmi'])
    plt.show()
    return


@app.cell
def _(df, mo):
    # We need this function for the trimmed mean
    from scipy.stats import trim_mean

    mean_val = df['bmi'].mean()
    trimmed_mean_val = trim_mean(df['bmi'], 0.1) 
    median_val = df['bmi'].median()

    # Measures of Dispersion (Spread)
    variance_val = df['bmi'].var()
    std_dev_val = df['bmi'].std()

    # Calculate Mean Absolute Deviation manually
    mean_abs_dev_val = (df['bmi'] - mean_val).abs().mean() 

    # Median Absolute Deviation (MAD) is calculated from the median
    median_abs_deviation_val = (df['bmi'] - median_val).abs().median()

    results_text = f"""
    ### Statistical Calculations for the Cleaned 'bmi' Column

    | Metric                      | Value                  |
    | --------------------------- | ---------------------- |
    | **Mean**                    | {mean_val:.4f}         |
    | **Trimmed Mean (10%)**      | {trimmed_mean_val:.4f} |
    | **Median**                  | {median_val:.4f}         |
    | **Variance**                | {variance_val:.4f}     |
    | **Standard Deviation**      | {std_dev_val:.4f}        |
    | **Mean Absolute Deviation** | {mean_abs_dev_val:.4f} |
    | **Median Absolute Deviation**| {median_abs_deviation_val:.4f}|
    """

    mo.md(results_text)
    return mean_val, results_text, std_dev_val


@app.cell
def _(df, mean_val, std_dev_val):
    # Min-Max Normalization
    # Formula: (x - min) / (max - min)
    min_val = df['bmi'].min()
    max_val = df['bmi'].max()
    df['bmi_min_max'] = (df['bmi'] - min_val) / (max_val - min_val)

    # Z-Score Normalization (Standardization)
    # Formula: (x - mean)
    df['bmi_z_score'] = (df['bmi'] - mean_val) / std_dev_val

    df[['bmi', 'bmi_min_max', 'bmi_z_score']].head()
    return


@app.cell
def _(results_text):
    # Writing results to file
    results_filename = 'statistical_results.txt'
    with open(results_filename, 'w') as f:
        f.write(results_text)
    print(f"Successfully saved the statistical results to {results_filename}")
    return


@app.cell
def _(df):
    # Writing modified dataframe to file
    output_filename = 'normalized_dataset.csv'

    df.to_csv(output_filename, index=False)

    print(f"Successfully saved the normalized data to {output_filename}")
    return


@app.cell
def _():
    # Create and save the metadata file programmatically because yes

    metadata_content = """
    # Метадані для лабораторної роботи з аналізу даних

    ## 1. Загальна інформація про набір даних

    - **Назва датасету:** Stroke Prediction Dataset (Прогнозування інсульту)
    - **Джерело:** Kaggle
    - **Пряме посилання:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
    - **Автор:** Fedesoriano

    ### Короткий опис

    Цей набір даних призначений для прогнозування ймовірності інсульту у пацієнтів на основі реальних медичних та демографічних показників. Він включає такі параметри, як стать, вік, наявність хронічних захворювань (гіпертонія, хвороби серця), спосіб життя (паління) та інші.

    ### Проведена обробка даних

    В рамках виконання завдання набір даних було очищено від пропущених значень та статистичних викидів.

    - **Початковий розмір:** 5110 рядків, 12 стовпців.
    - **Після видалення пропусків у стовпці `bmi`:** 4909 рядків.
    - **Фінальний розмір після видалення викидів у стовпці `bmi`:** 4799 рядків.

    Всі подальші розрахунки, нормалізація та візуалізація проводилися на повністю очищеному наборі даних.

    ## 2. Опис стовпців (Словник даних)

    Нижче наведено опис кожного стовпця в оригінальному наборі даних.

    | Назва стовпця      | Тип даних | Опис                                                                            |
    | ------------------ | --------- | ------------------------------------------------------------------------------- |
    | **id**             | `int64`   | Унікальний ідентифікатор пацієнта.                                              |
    | **gender**         | `object`  | Стать пацієнта: "Male" (Чоловік), "Female" (Жінка), "Other" (Інше).             |
    | **age**            | `float64` | Вік пацієнта.                                                                   |
    | **hypertension**   | `int64`   | Наявність гіпертонії у пацієнта (0 = Ні, 1 = Так).                                 |
    | **heart_disease**  | `int64`   | Наявність хвороб серця у пацієнта (0 = Ні, 1 = Так).                                |
    | **ever_married**   | `object`  | Чи був пацієнт коли-небудь одружений/заміжня ("Yes" або "No").                       |
    | **work_type**      | `object`  | Тип зайнятості ("Private", "Self-employed", "Govt_job", "children", "Never_worked"). |
    | **Residence_type** | `object`  | Тип місцевості проживання ("Urban" - місто, "Rural" - село).                       |
    | **avg_glucose_level**|`float64` | Середній рівень глюкози в крові.                                                 |
    | **bmi**            | `float64` | Індекс маси тіла.                                                               |
    | **smoking_status** | `object`  | Статус паління ("formerly smoked", "never smoked", "smokes", "Unknown").        |
    | **stroke**         | `int64`   | Чи був у пацієнта інсульт (1 = Так, 0 = Ні).                                     |
    """

    # md because i plan to upload this to github
    metadata_filename = 'metadata.md'

    with open(metadata_filename, 'w', encoding='utf-8') as ff:
        ff.write(metadata_content.strip())
    
    print(f"Файл з метаданими успішно створено та збережено як{metadata_filename}")
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df, plt, sns):
    # Age distribution for patients with and without stroke
    sns.violinplot(data=df, x='stroke', y='age')
    plt.title('Порівняння віку пацієнтів')
    plt.xlabel('Інсульт (0 = Ні, 1 = Так)')
    plt.ylabel('Вік')

    plt.xticks([0, 1], ['Ні', 'Так'])
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Relationship between BMI and Average Glucose Level
    sns.regplot(data=df, x='bmi', y='avg_glucose_level', line_kws={"color": "red"}, scatter_kws={"alpha": 0.3})
    plt.title('BMI vs. Рівень глюкози')
    plt.xlabel('Індекс маси тіла (BMI)')
    plt.ylabel('Середній рівень глюкози')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Graph 3: Stroke count by marital status :)
    sns.countplot(data=df, x='ever_married', hue='stroke')
    plt.title('Кількість інсультів в залежності від сімейного стану')
    plt.xlabel('Чи був коли-небудь одружений/заміжня')
    plt.ylabel('Кількість пацієнтів')
    plt.legend(title='Інсульт', labels=['Ні', 'Так'])
    plt.xticks([0, 1], ['Ні', 'Так'])
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Verification Graph: Age Distribution by Marital Status
    sns.boxplot(data=df, x='ever_married', y='age')

    plt.title('Порівняння віку одружених та неодружених пацієнтів')
    plt.xlabel('Чи був коли-небудь одружений/заміжня')
    plt.ylabel('Вік')

    plt.show()
    return


if __name__ == "__main__":
    app.run()
