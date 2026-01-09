"""
DFM Market Concentration Analysis Tool
A Streamlit application for analyzing Dubai Financial Market trading data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="DFM Market Concentration Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


class DFMDataParser:
    """Parser for DFM Yearly Bulletin Excel files"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.stocks = []
        self.raw_df = None
        self.parse_log = []
        
    def parse(self):
        """Parse the Excel file and extract stock data"""
        self.raw_df = pd.read_excel(self.file_path, sheet_name='Bulletins', header=None)
        current_sector = None
        
        for idx in range(2, len(self.raw_df)):
            row = self.raw_df.iloc[idx]
            col0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            
            # Stop at Market Trades Total
            if 'market trades total' in col0.lower():
                self.parse_log.append(f"Row {idx}: STOP - Market Trades Total")
                break
            
            # Skip Total rows
            if col0.lower().startswith('total') or 'grand total' in col0.lower():
                self.parse_log.append(f"Row {idx}: SKIP - Total row: {col0}")
                continue
            
            # Get trade value - col 12
            trade_val_raw = row.iloc[12]
            
            # Check if this is a sector header
            if pd.isna(trade_val_raw):
                if ' - ' not in col0 and col0:
                    current_sector = col0
                    self.parse_log.append(f"Row {idx}: SECTOR - {col0}")
                continue
            
            # Parse trade value
            try:
                if isinstance(trade_val_raw, str):
                    trade_val = float(trade_val_raw.replace(',', ''))
                elif np.isnan(trade_val_raw):
                    continue
                else:
                    trade_val = float(trade_val_raw)
            except:
                continue
                
            if trade_val == 0:
                continue
                
            # Parse other fields
            num_trades = self._parse_numeric(row.iloc[10])
            market_cap = self._parse_numeric(row.iloc[15])
            
            symbol = col0.split(' - ')[0].strip() if ' - ' in col0 else col0.strip()
            name = col0.split(' - ')[1].strip() if ' - ' in col0 else col0.strip()
            
            self.stocks.append({
                'symbol': symbol,
                'name': name,
                'trade_value': trade_val,
                'num_trades': num_trades,
                'market_cap': market_cap,
                'sector': current_sector,
                'row_idx': idx
            })
            self.parse_log.append(f"Row {idx}: STOCK - {symbol} (Sector: {current_sector})")
        
        return self.stocks
    
    def _parse_numeric(self, val):
        """Parse a numeric value from various formats"""
        if pd.isna(val):
            return 0
        if isinstance(val, str):
            try:
                return float(val.replace(',', ''))
            except:
                return 0
        try:
            return float(val)
        except:
            return 0
    
    def get_dataframe(self):
        """Convert parsed stocks to DataFrame"""
        if not self.stocks:
            self.parse()
        df = pd.DataFrame(self.stocks)
        df['avg_ticket'] = df['trade_value'] / df['num_trades']
        df['turnover_pct'] = (df['trade_value'] / df['market_cap']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df


class DFMAnalyzer:
    """Analyzer for DFM market concentration"""
    
    def __init__(self, df):
        self.df = df
        self.total_value = df['trade_value'].sum()
        self.total_trades = df['num_trades'].sum()
        self.num_stocks = len(df)
        self.avg_ticket = self.total_value / self.total_trades if self.total_trades > 0 else 0
        
    def get_market_overview(self):
        """Get market overview metrics"""
        return {
            'total_value': self.total_value,
            'total_trades': self.total_trades,
            'num_stocks': self.num_stocks,
            'avg_ticket': self.avg_ticket
        }
    
    def get_top_n(self, n=10):
        """Get top N stocks by value traded"""
        df = self.df.copy()
        df['pct_value'] = (df['trade_value'] / self.total_value) * 100
        df['pct_trades'] = (df['num_trades'] / self.total_trades) * 100
        df = df.sort_values('trade_value', ascending=False).head(n)
        df['rank'] = range(1, len(df) + 1)
        return df
    
    def get_bottom_n(self, n=10):
        """Get bottom N stocks by value traded (lowest performers)"""
        df = self.df.copy()
        df['pct_value'] = (df['trade_value'] / self.total_value) * 100
        df['pct_trades'] = (df['num_trades'] / self.total_trades) * 100
        df = df.sort_values('trade_value', ascending=True).head(n)
        df['rank'] = range(1, len(df) + 1)
        return df
    
    def calculate_hhi(self):
        """Calculate Herfindahl-Hirschman Index"""
        market_shares = (self.df['trade_value'] / self.total_value) * 100
        hhi = (market_shares ** 2).sum() / 10000  # Normalize to 0-1 scale
        effective_n = 1 / hhi if hhi > 0 else 0
        even_hhi = 1 / self.num_stocks if self.num_stocks > 0 else 0
        concentration_mult = hhi / even_hhi if even_hhi > 0 else 0
        
        return {
            'hhi': hhi,
            'effective_n': effective_n,
            'even_hhi': even_hhi,
            'concentration_mult': concentration_mult
        }
    
    def get_sector_breakdown(self):
        """Get sector-level aggregations"""
        sector_df = self.df.groupby('sector').agg({
            'trade_value': 'sum',
            'num_trades': 'sum',
            'symbol': 'count',
            'market_cap': 'sum'
        }).reset_index()
        sector_df.columns = ['sector', 'trade_value', 'num_trades', 'num_stocks', 'market_cap']
        sector_df['pct_value'] = (sector_df['trade_value'] / self.total_value) * 100
        sector_df = sector_df.sort_values('trade_value', ascending=False)
        return sector_df
    
    def get_ticket_distribution(self):
        """Get ticket size distribution statistics"""
        avg_tickets = self.df['avg_ticket'].dropna()
        return {
            'p25': avg_tickets.quantile(0.25),
            'median': avg_tickets.median(),
            'p75': avg_tickets.quantile(0.75),
            'mean': self.avg_ticket
        }


def format_value(val, unit='B'):
    """Format large numbers with units"""
    if unit == 'B':
        return f"AED {val/1e9:.2f}B"
    elif unit == 'M':
        return f"{val/1e6:.2f}M"
    elif unit == 'K':
        return f"AED {val/1000:.1f}K"
    return f"{val:,.0f}"


def create_trading_activity_chart(df, top_n_symbols):
    """Create scatter plot of trading activity - improved version matching expected format"""
    fig = go.Figure()
    
    # Other stocks (light blue, smaller)
    other = df[~df['symbol'].isin(top_n_symbols)]
    fig.add_trace(go.Scatter(
        x=other['num_trades'],
        y=other['avg_ticket'],
        mode='markers',
        name='Other listed securities',
        marker=dict(color='#a8d0e6', size=8, opacity=0.7),
        hovertemplate='<b>%{customdata[0]}</b><br>Trades: %{x:,.0f}<br>Avg Ticket: AED %{y:,.0f}<extra></extra>',
        customdata=other[['symbol']].values
    ))
    
    # Top 10 by Value Traded (orange, with smart label positioning)
    top = df[df['symbol'].isin(top_n_symbols)].copy()
    
    # Smart text positioning to avoid overlaps
    def get_text_position(symbol, num_trades, avg_ticket, all_top):
        # Position labels to minimize overlap based on location
        if symbol == 'EMAAR':
            return 'top right'
        elif symbol == 'SALIK':
            return 'bottom right'
        elif symbol in ['GULFNAV']:
            return 'top left'
        elif avg_ticket > 60000:
            return 'top center'
        else:
            return 'middle right'
    
    top['text_pos'] = top.apply(lambda r: get_text_position(r['symbol'], r['num_trades'], r['avg_ticket'], top), axis=1)
    
    fig.add_trace(go.Scatter(
        x=top['num_trades'],
        y=top['avg_ticket'],
        mode='markers+text',
        name='Top 10 by Value Traded',
        marker=dict(color='#d35400', size=12, line=dict(width=1, color='white')),
        text=top['symbol'],
        textposition=top['text_pos'].tolist(),
        textfont=dict(size=9, color='#2c3e50'),
        hovertemplate='<b>%{customdata[0]}</b><br>Trades: %{x:,.0f}<br>Avg Ticket: AED %{y:,.0f}<br>Value: AED %{customdata[1]:.2f}B<extra></extra>',
        customdata=np.column_stack([top['symbol'], top['trade_value']/1e9])
    ))
    
    # Calculate sensible y-axis range (cap at reasonable level to avoid outliers stretching chart)
    y_max = min(df['avg_ticket'].quantile(0.98) * 1.2, 250000)
    
    fig.update_layout(
        title=dict(
            text='DFM Listed (2025) — Average Trade Size vs Number of Trades<br><sup>(Top 10 by Value Traded highlighted)</sup>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Number of Trades (count)',
        yaxis_title='Average Trade Size (AED per trade) = Value Traded / # Trades',
        xaxis_type='log',
        xaxis=dict(
            tickvals=[100, 1000, 10000, 100000, 500000],
            ticktext=['100 trades', '1K trades', '10K trades', '100K trades', '500K trades'],
            gridcolor='#ecf0f1',
            showgrid=True
        ),
        yaxis=dict(
            range=[0, y_max],
            tickvals=[5000, 10000, 20000, 50000, 100000, 200000],
            ticktext=['AED 5K', 'AED 10K', 'AED 20K', 'AED 50K', 'AED 100K', 'AED 200K'],
            gridcolor='#ecf0f1',
            showgrid=True
        ),
        height=550,
        showlegend=True,
        legend=dict(
            yanchor="bottom", 
            y=0.02, 
            xanchor="right", 
            x=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_turnover_chart(df, top_n=10):
    """Create turnover velocity scatter plot"""
    df_valid = df[df['market_cap'] > 0].copy()
    df_valid['turnover_pct'] = (df_valid['trade_value'] / df_valid['market_cap']) * 100
    
    # Get top 10 by turnover
    top_turnover = df_valid.nlargest(top_n, 'turnover_pct')['symbol'].tolist()
    
    fig = go.Figure()
    
    # Other stocks
    other = df_valid[~df_valid['symbol'].isin(top_turnover)]
    fig.add_trace(go.Scatter(
        x=other['market_cap'],
        y=other['turnover_pct'],
        mode='markers',
        name='Other Listed Securities',
        marker=dict(color='#a8d0e6', size=10, opacity=0.6),
        hovertemplate='<b>%{customdata[0]}</b><br>Market Cap: AED %{x:.2e}<br>Turnover: %{y:.1f}%<extra></extra>',
        customdata=other[['symbol']].values
    ))
    
    # Top turnover drivers
    top = df_valid[df_valid['symbol'].isin(top_turnover)]
    fig.add_trace(go.Scatter(
        x=top['market_cap'],
        y=top['turnover_pct'],
        mode='markers+text',
        name='Top 10 Turnover Drivers',
        marker=dict(color='#f76c6c', size=14),
        text=top['symbol'],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{customdata[0]}</b><br>Market Cap: AED %{x:.2e}<br>Turnover: %{y:.1f}%<extra></extra>',
        customdata=top[['symbol']].values
    ))
    
    fig.update_layout(
        title='Turnover vs Market Cap (Listed, 2025) — Top 10 Turnover Drivers Highlighted',
        xaxis_title='Market Capitalization (AED)',
        yaxis_title='Turnover (%) = Value Traded / Market Cap',
        xaxis_type='log',
        yaxis_type='log',
        height=600,
        showlegend=True,
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    
    return fig


def create_retail_institutional_chart(df, top_n=10):
    """Create horizontal bar chart showing retail vs institutional profile"""
    df_sorted = df.nlargest(top_n * 3, 'trade_value').copy()
    df_sorted = df_sorted.sort_values('avg_ticket', ascending=True)
    
    def categorize(ticket):
        if ticket >= 60000:
            return 'Institutional-dominant'
        elif ticket >= 30000:
            return 'Mixed'
        else:
            return 'Retail-dominant'
    
    df_sorted['category'] = df_sorted['avg_ticket'].apply(categorize)
    
    colors = {
        'Institutional-dominant': '#f76c6c',
        'Mixed': '#667eea',
        'Retail-dominant': '#28a745'
    }
    
    fig = go.Figure()
    
    for cat in ['Retail-dominant', 'Mixed', 'Institutional-dominant']:
        cat_df = df_sorted[df_sorted['category'] == cat]
        fig.add_trace(go.Bar(
            y=cat_df['symbol'],
            x=cat_df['avg_ticket'],
            orientation='h',
            name=cat,
            marker_color=colors[cat],
            text=[f"Value: AED {v/1e9:.1f}B | Trades: {int(t):,}" for v, t in zip(cat_df['trade_value'], cat_df['num_trades'])],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Avg Ticket: AED %{x:,.0f}<br>%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Top Names by Value Traded — Retail vs Mixed vs Institutional (2025)',
        xaxis_title='Average Trade Size (AED) = Value Traded / # Trades',
        yaxis_title='',
        height=600,
        showlegend=True,
        barmode='stack',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(tickformat=',.0f')
    )
    
    return fig


def create_ticket_distribution_chart(df):
    """Create histogram showing distribution of stocks by average ticket size with classification zones"""
    fig = go.Figure()
    
    # Categorize stocks
    def categorize(ticket):
        if pd.isna(ticket) or ticket <= 0:
            return 'N/A'
        elif ticket >= 60000:
            return 'Institutional'
        elif ticket >= 30000:
            return 'Mixed'
        else:
            return 'Retail'
    
    df_valid = df[df['avg_ticket'].notna() & (df['avg_ticket'] > 0)].copy()
    df_valid['category'] = df_valid['avg_ticket'].apply(categorize)
    
    colors = {
        'Retail': '#28a745',
        'Mixed': '#667eea', 
        'Institutional': '#f76c6c'
    }
    
    # Create histogram bars for each category
    for cat in ['Retail', 'Mixed', 'Institutional']:
        cat_data = df_valid[df_valid['category'] == cat]['avg_ticket']
        if len(cat_data) > 0:
            fig.add_trace(go.Histogram(
                x=cat_data,
                name=f'{cat} ({len(cat_data)} stocks)',
                marker_color=colors[cat],
                opacity=0.8,
                xbins=dict(start=0, end=max(df_valid['avg_ticket']) * 1.1, size=10000),
                hovertemplate=f'{cat}<br>Ticket Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
    
    # Add threshold lines
    fig.add_vline(x=30000, line_dash="dash", line_color="gray", 
                  annotation_text="30K (Retail/Mixed)", annotation_position="top")
    fig.add_vline(x=60000, line_dash="dash", line_color="gray",
                  annotation_text="60K (Mixed/Institutional)", annotation_position="top")
    
    fig.update_layout(
        title='Distribution of Stocks by Average Ticket Size',
        xaxis_title='Average Ticket Size (AED)',
        yaxis_title='Number of Stocks',
        height=400,
        barmode='stack',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(tickformat=',.0f', tickvals=[0, 30000, 60000, 100000, 150000, 200000]),
        plot_bgcolor='white'
    )
    
    return fig


def create_ticket_category_summary_chart(df, total_value):
    """Create a summary chart showing stock count AND value contribution by category"""
    
    def categorize(ticket):
        if pd.isna(ticket) or ticket <= 0:
            return 'N/A'
        elif ticket >= 60000:
            return 'Institutional'
        elif ticket >= 30000:
            return 'Mixed'
        else:
            return 'Retail'
    
    df_valid = df[df['avg_ticket'].notna() & (df['avg_ticket'] > 0)].copy()
    df_valid['category'] = df_valid['avg_ticket'].apply(categorize)
    
    # Aggregate by category
    summary = df_valid.groupby('category').agg({
        'symbol': 'count',
        'trade_value': 'sum'
    }).reset_index()
    summary.columns = ['Category', 'Stock Count', 'Value Traded']
    summary['Value %'] = (summary['Value Traded'] / total_value) * 100
    
    # Sort in logical order
    cat_order = {'Retail': 0, 'Mixed': 1, 'Institutional': 2}
    summary['sort_order'] = summary['Category'].map(cat_order)
    summary = summary.sort_values('sort_order')
    
    colors = {'Retail': '#28a745', 'Mixed': '#667eea', 'Institutional': '#f76c6c'}
    
    # Create subplots - one for count, one for value
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("By Number of Stocks", "By Value Traded"))
    
    fig.add_trace(go.Pie(
        labels=summary['Category'],
        values=summary['Stock Count'],
        marker_colors=[colors[c] for c in summary['Category']],
        hole=0.4,
        textinfo='label+value',
        texttemplate='%{label}<br>%{value} stocks',
        hovertemplate='%{label}<br>%{value} stocks (%{percent})<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=summary['Category'],
        values=summary['Value %'],
        marker_colors=[colors[c] for c in summary['Category']],
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{percent} of value traded<extra></extra>'
    ), row=1, col=2)
    
    fig.update_layout(
        title='Market Composition: Stock Count vs Value Contribution by Investor Type',
        height=400,
        showlegend=False
    )
    
    return fig, summary


def create_sector_chart(sector_df):
    """Create sector breakdown pie chart"""
    fig = px.pie(
        sector_df,
        values='pct_value',
        names='sector',
        title='Sector Concentration by Value Traded (2025)',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(
        textposition='outside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Value: %{value:.1f}%<extra></extra>'
    )
    
    fig.update_layout(height=500)
    
    return fig


def main():
    st.title("📊 DFM Market Concentration Analysis")
    st.markdown("**Analyze Dubai Financial Market trading data for concentration patterns and liquidity insights**")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Input")
        uploaded_file = st.file_uploader(
            "Upload DFM Yearly Bulletin",
            type=['xlsx'],
            help="Upload the Yearly_Bulletin Excel file from DFM"
        )
        
        st.markdown("---")
        st.markdown("### Expected File Format")
        st.markdown("""
        - Sheet name: `Bulletins`
        - Contains sector headers and stock data
        - Columns: Symbol, Trade Value, No. of Trades, Market Cap
        """)
    
    if uploaded_file is not None:
        # Parse data
        with st.spinner("Parsing data..."):
            parser = DFMDataParser(uploaded_file)
            parser.parse()
            df = parser.get_dataframe()
            analyzer = DFMAnalyzer(df)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Market Overview",
            "🎯 HHI Concentration",
            "📊 Trading Activity",
            "🔄 Turnover Analysis",
            "👥 Investor Profile",
            "🔍 Debug & Validation"
        ])
        
        # Tab 1: Market Overview
        with tab1:
            overview = analyzer.get_market_overview()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value Traded", format_value(overview['total_value'], 'B'))
            with col2:
                st.metric("Total Trades", f"{overview['total_trades']/1e6:.2f}M")
            with col3:
                st.metric("Listed Stocks", overview['num_stocks'])
            with col4:
                st.metric("Avg Ticket Size", format_value(overview['avg_ticket'], 'K'))
            
            st.markdown("---")
            
            # Top 10 and Bottom 10 side by side
            col1, col2 = st.columns(2)
            
            top10 = analyzer.get_top_n(10)
            bottom10 = analyzer.get_bottom_n(10)
            
            with col1:
                st.subheader("🏆 Top 10 Stocks by Value Traded")
                st.caption("*Value Traded in AED Billions (B) · # Trades in Thousands (K) · Avg Ticket in AED Thousands (K)*")
                
                # Format the table with clean numbers
                display_top = top10[['rank', 'symbol', 'trade_value', 'pct_value', 'num_trades', 'pct_trades', 'avg_ticket']].copy()
                display_top.columns = ['Rank', 'Symbol', 'Value (B)', '% of Total', 'Trades (K)', '% Trades', 'Avg Ticket (K)']
                display_top['Value (B)'] = display_top['Value (B)'].apply(lambda x: f"{x/1e9:.2f}")
                display_top['% of Total'] = display_top['% of Total'].apply(lambda x: f"{x:.1f}%")
                display_top['Trades (K)'] = display_top['Trades (K)'].apply(lambda x: f"{x/1000:.1f}")
                display_top['% Trades'] = display_top['% Trades'].apply(lambda x: f"{x:.1f}%")
                display_top['Avg Ticket (K)'] = display_top['Avg Ticket (K)'].apply(lambda x: f"{x/1000:.1f}")
                
                st.dataframe(display_top, hide_index=True, use_container_width=True)
                
                # Key insights
                top10_pct = top10['pct_value'].sum()
                st.markdown(f"""
                <div class="success-box">
                    <strong>💰 Top 10 Impact:</strong> These stocks account for <strong>{top10_pct:.1f}%</strong> of total value traded. 
                    EMAAR alone contributes <strong>{top10['pct_value'].iloc[0]:.1f}%</strong>.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("⚠️ Bottom 10 Stocks by Value Traded")
                st.caption("*Value Traded in AED Millions (M) · # Trades · Avg Ticket in AED Thousands (K)*")
                
                # Format the table with clean numbers
                display_bottom = bottom10[['rank', 'symbol', 'trade_value', 'pct_value', 'num_trades', 'pct_trades', 'avg_ticket']].copy()
                display_bottom.columns = ['Rank', 'Symbol', 'Value (M)', '% of Total', 'Trades', '% Trades', 'Avg Ticket (K)']
                display_bottom['Value (M)'] = display_bottom['Value (M)'].apply(lambda x: f"{x/1e6:.2f}")
                display_bottom['% of Total'] = display_bottom['% of Total'].apply(lambda x: f"{x:.4f}%")
                display_bottom['Trades'] = display_bottom['Trades'].apply(lambda x: f"{int(x):,}")
                display_bottom['% Trades'] = display_bottom['% Trades'].apply(lambda x: f"{x:.4f}%")
                display_bottom['Avg Ticket (K)'] = display_bottom['Avg Ticket (K)'].apply(lambda x: f"{x/1000:.1f}" if pd.notna(x) and x > 0 else "N/A")
                
                st.dataframe(display_bottom, hide_index=True, use_container_width=True)
                
                bottom10_pct = bottom10['pct_value'].sum()
                st.markdown(f"""
                <div class="warning-box">
                    <strong>🎯 Attention Needed:</strong> Bottom 10 stocks contribute only <strong>{bottom10_pct:.3f}%</strong> of total value. 
                    These names may benefit from increased market making or investor awareness initiatives.
                </div>
                """, unsafe_allow_html=True)
            
            # Sector breakdown
            st.markdown("---")
            st.subheader("🏢 Sector Breakdown")
            
            sector_df = analyzer.get_sector_breakdown()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = create_sector_chart(sector_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.caption("*Value Traded in AED Billions (B)*")
                sector_display = sector_df[['sector', 'trade_value', 'pct_value', 'num_stocks']].copy()
                sector_display.columns = ['Sector', 'Value (B)', '% of Total', '# Stocks']
                sector_display['Value (B)'] = sector_display['Value (B)'].apply(lambda x: f"{x/1e9:.2f}")
                sector_display['% of Total'] = sector_display['% of Total'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(sector_display, hide_index=True, use_container_width=True)
                
                top3_pct = sector_df.head(3)['pct_value'].sum()
                st.markdown(f"""
                <div class="insight-box">
                    <strong>💡 Key Insight:</strong> Top 3 sectors (Real Estate, Financials, Industrials) 
                    account for <strong>{top3_pct:.1f}%</strong> of total value traded.
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 2: HHI Concentration
        with tab2:
            st.subheader("🎯 Herfindahl-Hirschman Index (HHI) Analysis")
            
            hhi_metrics = analyzer.calculate_hhi()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("HHI Index", f"{hhi_metrics['hhi']:.4f}")
            with col2:
                st.metric("Effective # of Stocks", f"~{hhi_metrics['effective_n']:.0f}")
            with col3:
                st.metric("Even Distribution HHI", f"{hhi_metrics['even_hhi']:.5f}")
            with col4:
                st.metric("Concentration Multiplier", f"{hhi_metrics['concentration_mult']:.1f}x")
            
            st.markdown("---")
            
            st.markdown(r"""
            ### Understanding HHI
            
            The **Herfindahl-Hirschman Index (HHI)** measures market concentration. It's calculated as the sum of squared market shares:
            
            $$HHI = \sum_{i=1}^{N} s_i^2$$
            
            Where $s_i$ is each stock's share of total value traded.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="insight-box">
                    <strong>📊 What the numbers mean:</strong>
                    <ul>
                        <li><strong>HHI = 0.114:</strong> Market is moderately to highly concentrated</li>
                        <li><strong>Effective N ≈ 9:</strong> Activity is concentrated into ~9 names</li>
                        <li><strong>7.5x multiplier:</strong> Market is 7.5x more concentrated than even distribution</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Key Implication:</strong>
                    <p>DFM's listed value traded in 2025 is as concentrated as if only 
                    <strong>~{hhi_metrics['effective_n']:.0f} names</strong> shared trading value equally.</p>
                    <p>This means market liquidity is materially dependent on a small core of stocks.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # HHI interpretation scale - based on Effective N (liquidity concentration)
            st.markdown("### Liquidity Concentration Scale (by Effective N)")
            
            st.markdown("""
            *Note: Standard DOJ/FTC HHI thresholds (0-10,000 scale) are designed for industry antitrust analysis, 
            not stock market liquidity. For trading/liquidity concentration, **Effective N** is the more meaningful metric.*
            """)
            
            hhi_scale = pd.DataFrame({
                'Effective N': ['< 10 stocks', '10 - 25 stocks', '25 - 50 stocks', '> 50 stocks'],
                'HHI Range': ['> 0.10', '0.04 - 0.10', '0.02 - 0.04', '< 0.02'],
                'Classification': ['🔴 Highly Concentrated', '🟠 Moderately Concentrated', '🟡 Low Concentration', '🟢 Well Distributed'],
                'DFM Status': [f'✓ DFM (~{hhi_metrics["effective_n"]:.0f} stocks)', '', '', '']
            })
            st.dataframe(hhi_scale, hide_index=True, use_container_width=True)
            
            st.markdown(f"""
            <div class="warning-box">
                <strong>🔴 DFM Classification: HIGHLY CONCENTRATED</strong>
                <p>With only <strong>~{hhi_metrics['effective_n']:.0f} effective stocks</strong> driving liquidity out of 66 listed securities, 
                DFM exhibits high trading concentration. The market is <strong>{hhi_metrics['concentration_mult']:.1f}x more concentrated</strong> 
                than an evenly distributed market would be.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tab 3: Trading Activity
        with tab3:
            st.subheader("📊 Trading Activity Analysis")
            st.markdown("*The primary driver of DFM's P&L through trading commissions*")
            
            # Get top and bottom stocks
            top10 = analyzer.get_top_n(10)
            bottom10 = analyzer.get_bottom_n(10)
            
            # Chart
            fig = create_trading_activity_chart(df, top10['symbol'].tolist())
            st.plotly_chart(fig, use_container_width=True)
            
            # Top names summary below chart (matching expected format)
            st.markdown("**Top names by value traded and their shares of listed value** *(Value in AED Billions)*")
            top5 = top10.head(5)
            for _, row in top5.iterrows():
                st.markdown(f"- **{row['symbol']}** ({row['trade_value']/1e9:.2f}, {row['pct_value']:.1f}%)")
            
            remaining = top10.iloc[5:]['symbol'].tolist()
            st.markdown(f"*(followed by {', '.join(remaining)} in the top 10)*")
            
            st.markdown("---")
            
            # Top 10 Table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top 10 by Value Traded")
                st.caption("*Value in AED Billions (B) · Trades in Thousands (K) · Avg Ticket in AED Thousands (K)*")
                
                display_top = top10[['rank', 'symbol', 'sector', 'trade_value', 'pct_value', 'num_trades', 'pct_trades', 'avg_ticket']].copy()
                display_top.columns = ['Rank', 'Symbol', 'Sector', 'Value (B)', '% of Total', 'Trades (K)', '% Trades', 'Avg Ticket (K)']
                display_top['Value (B)'] = display_top['Value (B)'].apply(lambda x: f"{x/1e9:.2f}")
                display_top['% of Total'] = display_top['% of Total'].apply(lambda x: f"{x:.1f}%")
                display_top['Trades (K)'] = display_top['Trades (K)'].apply(lambda x: f"{x/1000:.1f}")
                display_top['% Trades'] = display_top['% Trades'].apply(lambda x: f"{x:.1f}%")
                display_top['Avg Ticket (K)'] = display_top['Avg Ticket (K)'].apply(lambda x: f"{x/1000:.1f}")
                
                st.dataframe(display_top, hide_index=True, use_container_width=True)
                
                top10_value_pct = top10['pct_value'].sum()
                top10_trades_pct = top10['pct_trades'].sum()
                st.markdown(f"""
                <div class="success-box">
                    <strong>💰 Top 10 Impact:</strong> These stocks generate <strong>{top10_value_pct:.1f}%</strong> of total value traded 
                    and <strong>{top10_trades_pct:.1f}%</strong> of all trades. Focus resources here for maximum commission revenue.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ⚠️ Bottom 10 by Value Traded")
                st.caption("*Value in AED Millions (M) · Trades · Avg Ticket in AED Thousands (K)*")
                
                display_bottom = bottom10[['rank', 'symbol', 'sector', 'trade_value', 'pct_value', 'num_trades', 'pct_trades', 'avg_ticket']].copy()
                display_bottom.columns = ['Rank', 'Symbol', 'Sector', 'Value (M)', '% of Total', 'Trades', '% Trades', 'Avg Ticket (K)']
                display_bottom['Value (M)'] = display_bottom['Value (M)'].apply(lambda x: f"{x/1e6:.2f}")
                display_bottom['% of Total'] = display_bottom['% of Total'].apply(lambda x: f"{x:.4f}%")
                display_bottom['Trades'] = display_bottom['Trades'].apply(lambda x: f"{int(x):,}")
                display_bottom['% Trades'] = display_bottom['% Trades'].apply(lambda x: f"{x:.4f}%")
                display_bottom['Avg Ticket (K)'] = display_bottom['Avg Ticket (K)'].apply(lambda x: f"{x/1000:.1f}" if pd.notna(x) and x > 0 else "N/A")
                
                st.dataframe(display_bottom, hide_index=True, use_container_width=True)
                
                bottom10_value_pct = bottom10['pct_value'].sum()
                st.markdown(f"""
                <div class="warning-box">
                    <strong>🎯 Opportunity:</strong> Bottom 10 stocks contribute only <strong>{bottom10_value_pct:.3f}%</strong> of total value. 
                    Targeted initiatives (market making, investor awareness, analyst coverage) could unlock additional trading activity.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Key insights section
            st.markdown("### 💡 Key Insights for Trading Commission Strategy")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emaar_pct = top10[top10['symbol'] == 'EMAAR']['pct_value'].values[0]
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🏢 Single-Stock Risk:</strong>
                    <p>EMAAR alone accounts for <strong>{emaar_pct:.1f}%</strong> of all value traded. 
                    Commission revenue is heavily dependent on this one name.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Identify institutional vs retail stocks
                top10_institutional = top10[top10['avg_ticket'] >= 50000]['symbol'].tolist()
                top10_retail = top10[top10['avg_ticket'] < 35000]['symbol'].tolist()
                st.markdown(f"""
                <div class="insight-box">
                    <strong>👥 Client Mix:</strong>
                    <p><strong>Institutional-heavy:</strong> {', '.join(top10_institutional[:3]) if top10_institutional else 'None'}</p>
                    <p><strong>Retail-heavy:</strong> {', '.join(top10_retail) if top10_retail else 'None'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Sector concentration in top 10
                top10_sectors = top10.groupby('sector')['pct_value'].sum().sort_values(ascending=False)
                top_sector = top10_sectors.index[0]
                top_sector_pct = top10_sectors.values[0]
                st.markdown(f"""
                <div class="insight-box">
                    <strong>🏭 Sector Exposure:</strong>
                    <p>Top 10 is dominated by <strong>{top_sector}</strong> ({top_sector_pct:.1f}% of value). 
                    Diversifying trading activity across sectors reduces concentration risk.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 4: Turnover Analysis
        with tab4:
            st.subheader("🔄 Turnover Velocity Analysis")
            
            fig = create_turnover_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <strong>📈 Understanding Turnover:</strong>
                <p><strong>Turnover = Value Traded / Market Cap × 100%</strong></p>
                <ul>
                    <li>High turnover indicates aggressive rotation relative to company size</li>
                    <li>Turnover leaders are NOT always the largest market cap names</li>
                    <li>Smaller/mid-cap names can show higher turnover ratios</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Prepare turnover data
            df_valid = df[df['market_cap'] > 0].copy()
            df_valid['turnover_pct'] = (df_valid['trade_value'] / df_valid['market_cap']) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top 10 by Turnover Velocity")
                st.caption("*Value & Market Cap in AED Billions (B) or Millions (M)*")
                
                top_turnover = df_valid.nlargest(10, 'turnover_pct')[['symbol', 'sector', 'trade_value', 'market_cap', 'turnover_pct']].copy()
                top_turnover['rank'] = range(1, len(top_turnover) + 1)
                
                turnover_top_display = top_turnover[['rank', 'symbol', 'sector', 'trade_value', 'market_cap', 'turnover_pct']].copy()
                turnover_top_display.columns = ['Rank', 'Symbol', 'Sector', 'Value (B)', 'Mkt Cap (B)', 'Turnover %']
                turnover_top_display['Value (B)'] = turnover_top_display['Value (B)'].apply(lambda x: f"{x/1e9:.2f}" if x >= 1e9 else f"{x/1e6:.0f}M")
                turnover_top_display['Mkt Cap (B)'] = turnover_top_display['Mkt Cap (B)'].apply(lambda x: f"{x/1e9:.2f}" if x >= 1e9 else f"{x/1e6:.0f}M")
                turnover_top_display['Turnover %'] = turnover_top_display['Turnover %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(turnover_top_display, hide_index=True, use_container_width=True)
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>🔥 High Velocity Names:</strong> These stocks show aggressive rotation relative to their size. 
                    High turnover can indicate strong investor interest or speculative activity.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ⚠️ Bottom 10 by Turnover Velocity")
                st.caption("*Value & Market Cap in AED Billions (B) or Millions (M)*")
                
                bottom_turnover = df_valid.nsmallest(10, 'turnover_pct')[['symbol', 'sector', 'trade_value', 'market_cap', 'turnover_pct']].copy()
                bottom_turnover['rank'] = range(1, len(bottom_turnover) + 1)
                
                turnover_bottom_display = bottom_turnover[['rank', 'symbol', 'sector', 'trade_value', 'market_cap', 'turnover_pct']].copy()
                turnover_bottom_display.columns = ['Rank', 'Symbol', 'Sector', 'Value (M)', 'Mkt Cap (B)', 'Turnover %']
                turnover_bottom_display['Value (M)'] = turnover_bottom_display['Value (M)'].apply(lambda x: f"{x/1e6:.2f}")
                turnover_bottom_display['Mkt Cap (B)'] = turnover_bottom_display['Mkt Cap (B)'].apply(lambda x: f"{x/1e9:.2f}" if x >= 1e9 else f"{x/1e6:.0f}M")
                turnover_bottom_display['Turnover %'] = turnover_bottom_display['Turnover %'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(turnover_bottom_display, hide_index=True, use_container_width=True)
                
                st.markdown(f"""
                <div class="warning-box">
                    <strong>🎯 Illiquid Names:</strong> Low turnover may indicate: limited investor awareness, 
                    tight shareholder base, or need for market making support. These represent opportunities 
                    to unlock additional trading activity.
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 5: Investor Profile
        with tab5:
            st.subheader("👥 Retail vs Institutional Profile")
            
            fig = create_retail_institutional_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Ticket size distribution
            st.markdown("---")
            st.subheader("📊 Ticket Size Distribution")
            
            dist = analyzer.get_ticket_distribution()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("25th Percentile", f"AED {dist['p25']/1000:.1f}K")
            with col2:
                st.metric("Median (50th)", f"AED {dist['median']/1000:.1f}K")
            with col3:
                st.metric("75th Percentile", f"AED {dist['p75']/1000:.1f}K")
            with col4:
                st.metric("Market Average", f"AED {dist['mean']/1000:.1f}K")
            
            # New histogram chart
            fig_hist = create_ticket_distribution_chart(df)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.markdown("---")
            
            # New pie charts showing count vs value
            st.subheader("📊 Market Composition Analysis")
            st.markdown("*Comparing number of stocks vs their contribution to total value traded*")
            
            fig_pies, category_summary = create_ticket_category_summary_chart(df, analyzer.total_value)
            st.plotly_chart(fig_pies, use_container_width=True)
            
            # Show the summary table
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Category Breakdown")
                st.caption("*Value in AED Billions (B)*")
                summary_display = category_summary[['Category', 'Stock Count', 'Value Traded', 'Value %']].copy()
                summary_display['Value (B)'] = summary_display['Value Traded'].apply(lambda x: f"{x/1e9:.2f}")
                summary_display['Value %'] = summary_display['Value %'].apply(lambda x: f"{x:.1f}%")
                summary_display = summary_display[['Category', 'Stock Count', 'Value (B)', 'Value %']]
                st.dataframe(summary_display, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                    <strong>💡 Key Insight:</strong>
                    <p>While Retail-dominant stocks may outnumber Institutional stocks, 
                    the <strong>value contribution</strong> tells a different story.</p>
                    <p>A few large institutional names drive the majority of trading value, 
                    even if more stocks cater to retail investors.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                    <strong>📋 Classification Thresholds:</strong>
                    <ul>
                        <li><strong style="color:#f76c6c">● Institutional:</strong> Avg Ticket > AED 60K</li>
                        <li><strong style="color:#667eea">● Mixed:</strong> Avg Ticket AED 30K - 60K</li>
                        <li><strong style="color:#28a745">● Retail:</strong> Avg Ticket < AED 30K</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 6: Debug & Validation
        with tab6:
            st.subheader("🔍 Data Validation & Debug")
            
            # Validation metrics
            st.markdown("### Validation Against Expected Values")
            
            expected = {
                'Total Value Traded': (165.2e9, overview['total_value']),
                'Total Trades': (3.35e6, overview['total_trades']),
                'Avg Ticket Size': (49300, overview['avg_ticket']),
                'Number of Stocks': (66, overview['num_stocks'])
            }
            
            val_data = []
            for metric, (exp, act) in expected.items():
                diff_pct = abs(act - exp) / exp * 100
                status = "✅" if diff_pct < 5 else "⚠️"
                val_data.append({
                    'Metric': metric,
                    'Expected': f"{exp:,.0f}" if exp > 1000 else f"{exp:.0f}",
                    'Actual': f"{act:,.0f}" if act > 1000 else f"{act:.0f}",
                    'Difference': f"{diff_pct:.2f}%",
                    'Status': status
                })
            
            st.dataframe(pd.DataFrame(val_data), hide_index=True, use_container_width=True)
            
            # Raw data preview
            with st.expander("📋 Raw Data Preview (First 20 Rows)"):
                st.dataframe(parser.raw_df.head(20))
            
            # Parsed stocks
            with st.expander("📊 Parsed Stock Data"):
                st.dataframe(df[['symbol', 'name', 'sector', 'trade_value', 'num_trades', 'market_cap', 'avg_ticket']])
            
            # Parse log
            with st.expander("📝 Parsing Log"):
                st.text("\n".join(parser.parse_log[:50]))
                if len(parser.parse_log) > 50:
                    st.text(f"... and {len(parser.parse_log) - 50} more entries")
    
    else:
        # No file uploaded - show instructions
        st.info("👈 Please upload a DFM Yearly Bulletin Excel file to begin analysis")
        
        st.markdown("""
        ### About This Tool
        
        This application analyzes Dubai Financial Market (DFM) trading data to provide insights on:
        
        - **Market Concentration**: HHI analysis showing how concentrated trading activity is
        - **Top Performers**: Rankings of stocks by value traded and trade count
        - **Sector Analysis**: Breakdown of activity by market sector
        - **Investor Profile**: Analysis of retail vs institutional participation patterns
        - **Turnover Velocity**: Which stocks "rotate" most relative to their size
        
        ### Expected Results
        
        For the 2025 DFM Yearly Bulletin, you should see:
        - **Total Value Traded**: ~AED 165.2B
        - **Total Trades**: ~3.35M
        - **66 Listed Securities**
        - **Top stock (EMAAR)**: ~28% of total value
        - **HHI Index**: ~0.114 (moderately concentrated)
        """)


if __name__ == "__main__":
    main()