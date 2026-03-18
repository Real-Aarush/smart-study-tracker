import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    df = pd.read_csv("data/student_data.csv")

    X = df[['study_hours', 'sleep_hours', 'attendance', 'prev_score']]
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model=LinearRegression()
    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    r2  = r2_score(y_test, y_predict)

    return model, mae, r2