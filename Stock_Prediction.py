import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle

# Company names
companies = [
    "Adani Ports and Special Economic Zone Ltd.", "Asian Paints Ltd.", "Axis Bank Ltd.", "Bajaj Auto Ltd.",
    "Bajaj Finserv Ltd.", "Bajaj Finance Ltd.", "Bharti Airtel Ltd.", "Bharat Petroleum Corporation Ltd.",
    "Britannia Industries Ltd.", "Cipla Ltd.", "Coal India Ltd.", "Dr. Reddy's Laboratories Ltd.", "Eicher Motors Ltd.",
    "GAIL (India) Ltd.", "Grasim Industries Ltd.", "HCL Technologies Ltd.", "Housing Development Finance Corporation Ltd.",
    "HDFC Bank Ltd.", "Hero MotoCorp Ltd.", "Hindalco Industries Ltd.", "Hindustan Unilever Ltd.", "ICICI Bank Ltd.",
    "IndusInd Bank Ltd.", "Bharti Infratel Ltd.", "Infosys Ltd.", "Indian Oil Corporation Ltd.", "ITC Ltd.",
    "JSW Steel Ltd.", "Kotak Mahindra Bank Ltd.", "Larsen & Toubro Ltd.", "Mahindra & Mahindra Ltd.", "Maruti Suzuki India Ltd.",
    "Nestle India Ltd.", "NTPC Ltd.", "Oil & Natural Gas Corporation Ltd.", "Power Grid Corporation of India Ltd.",
    "Reliance Industries Ltd.", "State Bank of India", "Shree Cement Ltd.", "Sun Pharmaceutical Industries Ltd.",
    "Tata Motors Ltd.", "Tata Steel Ltd.", "Tata Consultancy Services Ltd.", "Tech Mahindra Ltd.", "Titan Company Ltd.",
    "UltraTech Cement Ltd.", "UPL Ltd.", "Vedanta Ltd.", "Wipro Ltd.", "Zee Entertainment Enterprises Ltd."
]

# Corresponding CSV file names
files = [
    "ADANIPORTS.csv", "ASIANPAINT.csv", "AXISBANK.csv", "BAJAJ-AUTO.csv", "BAJAJFINSV.csv", "BAJFINANCE.csv",
    "BHARTIARTL.csv", "BPCL.csv", "BRITANNIA.csv", "CIPLA.csv", "COALINDIA.csv", "DRREDDY.csv", "EICHERMOT.csv",
    "GAIL.csv", "GRASIM.csv", "HCLTECH.csv", "HDFC.csv", "HDFCBANK.csv", "HEROMOTOCO.csv", "HINDALCO.csv",
    "HINDUNILVR.csv", "ICICIBANK.csv", "INDUSINDBK.csv", "INFRATEL.csv", "INFY.csv", "IOC.csv", "ITC.csv",
    "JSWSTEEL.csv", "KOTAKBANK.csv", "LT.csv", "MM.csv", "MARUTI.csv", "NESTLEIND.csv", "NTPC.csv", "ONGC.csv",
    "POWERGRID.csv", "RELIANCE.csv", "SBIN.csv", "SHREECEM.csv", "SUNPHARMA.csv", "TATAMOTORS.csv", "TATASTEEL.csv",
    "TCS.csv", "TECHM.csv", "TITAN.csv", "ULTRACEMCO.csv", "UPL.csv", "VEDL.csv", "WIPRO.csv", "ZEEL.csv"
]

# Printing company names with serial numbers
print("Please choose a company to train the model on:\n")
for i, company in enumerate(companies, 1):
    print(f"{i}. {company}")

# Asking the user for their choice
choice = int(input("\nEnter the serial number of the company: "))

if 1 <= choice <= len(companies):
    selected_file = files[choice - 1]
    company_name = companies[choice - 1]
    print(f"\nYou selected {company_name}, training model on {selected_file}...\n")

    # Loading the data
    df = pd.read_csv(f"dataset\{selected_file}")

    # Parsing date and setting index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Droping unnecessary columns
    df = df.drop(columns=['symbol', 'series'], errors='ignore')

    # Feature Engineering
    n = 5
    df_features = pd.DataFrame()

    for i in range(1, n + 1):
        df_features[f'lag_{i}'] = df['Close'].shift(i)

    df_features = df_features.dropna()

    X = df_features
    y = df['Close'][n:]

    # Train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Training the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100 
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")  

    # Saving the model
    model_filename = f"random_forest_{company_name.replace(' ', '_').replace('&', 'and')}pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved as {model_filename}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='orange')
    plt.title(f'{company_name} - Actual vs Predicted Closing Prices')
    plt.xlabel('Year')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("Invalid choice! Please choose a number between 1 and", len(companies))
