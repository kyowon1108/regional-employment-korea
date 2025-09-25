#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Employment Rate Analysis Module
Main analysis module for employment rate analysis using integrated data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Set up paths - updated for new folder structure
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Korean font settings
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class EmploymentAnalyzer:
    """Employment rate analysis class"""
    
    def __init__(self, data_file="최종_통합데이터_수정_utf-8.csv"):
        """Initialize analyzer with data file"""
        self.data_file = DATA_DIR / data_file
        self.df_yearly = None
        self.analysis_results = None
        
    def load_and_examine_data(self):
        """Load and examine data"""
        print("=== Data Loading and Basic Information ===")
        
        # Load data
        df1 = pd.read_csv(self.data_file)
        print(f"Data shape: {df1.shape}")
        
        print("\nColumns:")
        for i, col in enumerate(df1.columns):
            print(f"{i+1:2d}. {col}")
        
        print(f"\n=== Basic Data Information ===")
        print(f"Total rows: {len(df1):,}")
        print(f"Total columns: {len(df1.columns)}")
        print(f"Year range: {df1['연도'].min()} ~ {df1['연도'].max()}")
        print(f"Provinces: {df1['시도'].nunique()}")
        print(f"Counties: {df1['시군구'].nunique()}")
        
        print(f"\n=== Province Distribution ===")
        print(df1['시도'].value_counts().head(10))
        
        print(f"\n=== Year Distribution ===")
        print(df1['연도'].value_counts().sort_index())
        
        print(f"\n=== Half-year Distribution ===")
        print(df1['반기'].value_counts().sort_index())
        
        return df1
    
    def check_data_quality(self, df1):
        """Check data quality"""
        print("\n=== Data Quality Check ===")
        
        # Missing value check
        print("=== Missing Value Check ===")
        missing_data = df1.isnull().sum()
        missing_percent = (missing_data / len(df1)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Rate (%)': missing_percent
        })
        print(missing_df)
        
        # Employment rate distribution
        print(f"\n=== Employment Rate Distribution ===")
        print(f"Rows with 0 employment rate: {(df1['고용률'] == 0).sum()}")
        print(f"Rate of 0 employment rate: {((df1['고용률'] == 0).sum() / len(df1)) * 100:.1f}%")
        
        print(f"\n=== Employment Rate Statistics (excluding 0) ===")
        employment_rate_stats = df1[df1['고용률'] > 0]['고용률'].describe()
        print(employment_rate_stats)
    
    def preprocess_data(self, df1):
        """Data preprocessing and optimization"""
        print("\n=== Data Preprocessing and Optimization ===")
        
        # 1. Exclude COVID-19 period (2021 onwards only)
        df_post_covid = df1[df1['연도'] >= 2021].copy()
        print(f"COVID-19 period excluded: {df1.shape[0]:,} rows → {df_post_covid.shape[0]:,} rows")
        
        # 2. Replace 0 employment rates with NaN
        df_post_covid['고용률'] = df_post_covid['고용률'].replace(0, np.nan)
        df_post_covid['취업자'] = df_post_covid['취업자'].replace(0, np.nan)
        
        # 3. Remove rows with missing employment rates
        df = df_post_covid.dropna(subset=['고용률']).copy()
        print(f"Missing employment rate removed: {df_post_covid.shape[0]:,} rows → {df.shape[0]:,} rows")
        
        # 4. Annual aggregation (half-year to annual)
        df_yearly = df.groupby(['시도', '시군구', '연도']).agg({
            '고용률': 'mean',
            'E9_체류자수': 'sum',
            '제조업_종사자수': 'sum',
            '서비스업_종사자수': 'sum',
            '면적': 'first'
        }).reset_index()
        
        print(f"Annual aggregation: {df.shape[0]:,} rows → {df_yearly.shape[0]:,} rows")
        
        # 5. Create new variables
        df_yearly['E9_density'] = df_yearly['E9_체류자수'] / df_yearly['면적']
        df_yearly['manufacturing_concentration'] = df_yearly['제조업_종사자수'] / (df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수'])
        df_yearly['service_concentration'] = df_yearly['서비스업_종사자수'] / (df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수'])
        df_yearly['total_employees'] = df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수']
        df_yearly['employee_density'] = df_yearly['total_employees'] / df_yearly['면적']
        
        # 6. Regional characteristic variables
        df_yearly['capital_area'] = df_yearly['시도'].isin(['서울특별시', '경기도', '인천광역시']).astype(int)
        df_yearly['manufacturing_center'] = (df_yearly['manufacturing_concentration'] > 0.5).astype(int)
        df_yearly['service_center'] = (df_yearly['service_concentration'] > 0.5).astype(int)
        
        # 7. Log transformation
        df_yearly['E9_density_log'] = np.log1p(df_yearly['E9_density'])
        df_yearly['employee_density_log'] = np.log1p(df_yearly['employee_density'])
        
        print(f"Final data: {df_yearly.shape[0]:,} rows, {df_yearly.shape[1]:,} columns")
        
        self.df_yearly = df_yearly
        return df_yearly
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        print("\n=== Comprehensive Analysis ===")
        
        # 1. Correlation analysis
        print("\n1. Correlation Analysis")
        correlation_vars = ['고용률', 'E9_density_log', 'manufacturing_concentration', 'employee_density_log', 'capital_area']
        correlation_matrix = self.df_yearly[correlation_vars].corr()
        print(correlation_matrix)
        
        # 2. Simple regression analysis
        print("\n2. Simple Regression Analysis")
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Basic model
        X_basic = self.df_yearly[['E9_density_log', 'manufacturing_concentration', 'employee_density_log']]
        y = self.df_yearly['고용률']
        
        model_basic = LinearRegression()
        model_basic.fit(X_basic, y)
        y_pred_basic = model_basic.predict(X_basic)
        
        print(f"Basic model R²: {r2_score(y, y_pred_basic):.4f}")
        print(f"Basic model RMSE: {np.sqrt(mean_squared_error(y, y_pred_basic)):.4f}")
        
        # 3. Regional analysis
        print("\n3. Regional Analysis")
        
        # Capital vs non-capital
        capital = self.df_yearly[self.df_yearly['capital_area'] == 1]
        non_capital = self.df_yearly[self.df_yearly['capital_area'] == 0]
        
        print(f"Capital area: {len(capital)} regions, avg employment rate: {capital['고용률'].mean():.2f}%")
        print(f"Non-capital area: {len(non_capital)} regions, avg employment rate: {non_capital['고용률'].mean():.2f}%")
        
        # Manufacturing vs service centers
        manufacturing = self.df_yearly[self.df_yearly['manufacturing_center'] == 1]
        service = self.df_yearly[self.df_yearly['service_center'] == 1]
        
        print(f"Manufacturing centers: {len(manufacturing)} regions, avg employment rate: {manufacturing['고용률'].mean():.2f}%")
        print(f"Service centers: {len(service)} regions, avg employment rate: {service['고용률'].mean():.2f}%")
        
        # 4. Time series analysis
        print("\n4. Time Series Analysis")
        yearly_avg = self.df_yearly.groupby('연도')['고용률'].mean()
        print("Annual average employment rate:")
        for year, rate in yearly_avg.items():
            print(f"  {year}: {rate:.2f}%")
        
        # 5. Statistical tests
        print("\n5. Statistical Tests")
        from scipy import stats
        
        # Capital vs non-capital employment rate difference test
        t_stat, p_value = stats.ttest_ind(capital['고용률'], non_capital['고용률'])
        print(f"Capital vs non-capital employment rate difference test: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Manufacturing vs service centers employment rate difference test
        t_stat, p_value = stats.ttest_ind(manufacturing['고용률'], service['고용률'])
        print(f"Manufacturing vs service centers employment rate difference test: t={t_stat:.4f}, p={p_value:.4f}")
        
        self.analysis_results = {
            'correlation': correlation_matrix,
            'model_basic': model_basic,
            'capital': capital,
            'non_capital': non_capital,
            'manufacturing': manufacturing,
            'service': service,
            'yearly_avg': yearly_avg
        }
        
        return self.analysis_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== Creating Visualizations ===")
        
        # 1. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_vars = ['고용률', 'E9_density_log', 'manufacturing_concentration', 'employee_density_log', 'capital_area']
        correlation_matrix = self.df_yearly[correlation_vars].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Variable Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Regional employment rate comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Capital vs non-capital
        capital = self.analysis_results['capital']
        non_capital = self.analysis_results['non_capital']
        
        axes[0, 0].boxplot([capital['고용률'], non_capital['고용률']], 
                           labels=['Capital Area', 'Non-Capital Area'])
        axes[0, 0].set_title('Capital vs Non-Capital Employment Rate Comparison')
        axes[0, 0].set_ylabel('Employment Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Manufacturing vs service centers
        manufacturing = self.analysis_results['manufacturing']
        service = self.analysis_results['service']
        
        axes[0, 1].boxplot([manufacturing['고용률'], service['고용률']], 
                           labels=['Manufacturing Centers', 'Service Centers'])
        axes[0, 1].set_title('Industry-based Employment Rate Comparison')
        axes[0, 1].set_ylabel('Employment Rate (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # E9 density vs employment rate
        axes[1, 0].scatter(self.df_yearly['E9_density_log'], self.df_yearly['고용률'], alpha=0.6, s=30)
        axes[1, 0].set_xlabel('E9 Resident Density (log)')
        axes[1, 0].set_ylabel('Employment Rate (%)')
        axes[1, 0].set_title('E9 Resident Density vs Employment Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Manufacturing concentration vs employment rate
        axes[1, 1].scatter(self.df_yearly['manufacturing_concentration'], self.df_yearly['고용률'], alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Manufacturing Concentration')
        axes[1, 1].set_ylabel('Employment Rate (%)')
        axes[1, 1].set_title('Manufacturing Concentration vs Employment Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'regional_industry_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Time series analysis
        plt.figure(figsize=(12, 6))
        
        # Annual average employment rate
        yearly_avg = self.analysis_results['yearly_avg']
        plt.subplot(1, 2, 1)
        plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Year')
        plt.ylabel('Average Employment Rate (%)')
        plt.title('National Annual Average Employment Rate Trend')
        plt.grid(True, alpha=0.3)
        
        # Regional annual employment rate
        plt.subplot(1, 2, 2)
        capital_yearly = self.df_yearly[self.df_yearly['capital_area'] == 1].groupby('연도')['고용률'].mean()
        non_capital_yearly = self.df_yearly[self.df_yearly['capital_area'] == 0].groupby('연도')['고용률'].mean()
        
        plt.plot(capital_yearly.index, capital_yearly.values, marker='o', label='Capital Area', linewidth=2)
        plt.plot(non_capital_yearly.index, non_capital_yearly.values, marker='s', label='Non-Capital Area', linewidth=2)
        plt.xlabel('Year')
        plt.ylabel('Average Employment Rate (%)')
        plt.title('Regional Annual Employment Rate Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization files created:")
        print("- correlation_heatmap.png")
        print("- regional_industry_analysis.png")
        print("- time_series_analysis.png")
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n=== Analysis Summary Report ===")
        print(f"Total analyzed regions: {len(self.df_yearly)}")
        print(f"Capital area regions: {len(self.analysis_results['capital'])}")
        print(f"Non-capital area regions: {len(self.analysis_results['non_capital'])}")
        print(f"Manufacturing center regions: {len(self.analysis_results['manufacturing'])}")
        print(f"Service center regions: {len(self.analysis_results['service'])}")
        
        # Key findings
        print("\n=== Key Findings ===")
        print(f"1. E9 density vs Employment rate correlation: {self.analysis_results['correlation'].loc['고용률', 'E9_density_log']:.4f}")
        print(f"2. Employee density vs Employment rate correlation: {self.analysis_results['correlation'].loc['고용률', 'employee_density_log']:.4f}")
        print(f"3. Capital area vs Non-capital area employment rate difference: {self.analysis_results['non_capital']['고용률'].mean() - self.analysis_results['capital']['고용률'].mean():.2f}%p")
        
        return {
            'total_regions': len(self.df_yearly),
            'capital_regions': len(self.analysis_results['capital']),
            'non_capital_regions': len(self.analysis_results['non_capital']),
            'manufacturing_regions': len(self.analysis_results['manufacturing']),
            'service_regions': len(self.analysis_results['service'])
        }

def main():
    """Main execution function"""
    print("Employment Rate Analysis - Integrated Data (Optimized Version)")
    print("=" * 70)
    
    try:
        # Initialize analyzer
        analyzer = EmploymentAnalyzer()
        
        # 1. Load and examine data
        df1 = analyzer.load_and_examine_data()
        
        # 2. Check data quality
        analyzer.check_data_quality(df1)
        
        # 3. Preprocess and optimize data
        df_yearly = analyzer.preprocess_data(df1)
        
        # 4. Run comprehensive analysis
        analysis_results = analyzer.run_analysis()
        
        # 5. Create visualizations
        analyzer.create_visualizations()
        
        # 6. Generate summary report
        summary = analyzer.generate_summary_report()
        
        print("\nOptimized analysis completed successfully!")
        print("Generated files:")
        print("- correlation_heatmap.png")
        print("- regional_industry_analysis.png")
        print("- time_series_analysis.png")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
