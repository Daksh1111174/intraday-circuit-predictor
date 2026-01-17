from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = [
    'pct_change',
    'volume_spike',
    'vwap_dist',
    'volatility'
]

def train_intraday_model(df):
    X = df[FEATURES]
    y = df['circuit_target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
