# DFM Market Concentration Analysis Tool

A Streamlit web application for analyzing Dubai Financial Market (DFM) trading data and generating comprehensive market concentration analysis reports.

## Features

### 📈 Market Overview
- Total Value Traded, Total Trades, Listed Stocks count
- Average Ticket Size calculation
- Top 10 stocks by value traded with detailed metrics
- Sector breakdown with visualization

### 🎯 HHI Concentration Analysis
- Herfindahl-Hirschman Index calculation
- Effective number of stocks driving liquidity
- Concentration multiplier vs even distribution
- Visual interpretation scale

### 📊 Trading Activity Charts
- Scatter plot: Number of Trades vs Average Ticket Size
- Logarithmic scale for better visualization
- Top 10 stocks highlighted with labels

### 🔄 Turnover Analysis
- Turnover velocity (Value Traded / Market Cap)
- Identifies stocks that "rotate" most aggressively
- Top 10 turnover drivers highlighted

### 👥 Investor Profile
- Retail vs Mixed vs Institutional classification
- Horizontal bar chart showing ticket size distribution
- Box plot of average ticket sizes across all stocks
- Quartile statistics (25th, 50th, 75th percentiles)

### 🔍 Debug & Validation
- Data validation against expected values
- Raw data preview
- Parsed stock data table
- Parsing log for troubleshooting

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run dfm_analysis_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Upload the DFM Yearly Bulletin Excel file (`Yearly_Bulletin_01012025.xlsx`)
2. Navigate through the tabs to explore different analyses:
   - **Market Overview**: Summary metrics and top performers
   - **HHI Concentration**: Market concentration analysis
   - **Trading Activity**: Visual analysis of trade patterns
   - **Turnover Analysis**: Stock rotation metrics
   - **Investor Profile**: Retail vs institutional breakdown
   - **Debug & Validation**: Data verification tools

## Expected Input File Format

- **File type**: Excel (.xlsx)
- **Sheet name**: `Bulletins`
- **Structure**: Sector headers followed by stock data
- **Key columns**:
  - Column 0: Symbol-Security Name
  - Column 10: No. of Trades
  - Column 12: Trade Value
  - Column 15: Market Capitalization

## Validation Targets

The application validates against these expected values:
- ✅ Total Value Traded: AED 165.2B
- ✅ Total Trades: 3.35M
- ✅ Number of Stocks: 66
- ✅ Average Ticket Size: AED 49.3K
- ✅ HHI Index: ~0.114

## Key Insights

Based on 2025 DFM data:
- **Top 10 stocks account for 77.7%** of total value traded
- **EMAAR alone contributes 28.1%** of market value
- **Market is 7.5x more concentrated** than even distribution
- **Effective liquidity core = ~9 stocks**
- **Top 3 sectors (Real Estate, Financials, Industrials) = 83.5%** of value

## Technical Stack

- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **openpyxl**: Excel file reading

## File Structure

```
├── dfm_analysis_app.py    # Main application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## License

For professional financial market analysis use.

## Author

Built for DFM market analysts to analyze trading concentration and liquidity patterns.
