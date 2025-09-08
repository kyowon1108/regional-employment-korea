#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel Data Analysis Module
패널분석을 통한 고용률 영향 요인 분석 (Fixed Effect, Random Effect, Two-way Fixed Effect)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel.results import PanelResults
from scipy import stats

warnings.filterwarnings('ignore')

# Set up paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Korean font settings
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class PanelAnalyzer:
    """패널분석 클래스"""
    
    def __init__(self, data_file="최종_통합데이터_수정_utf-8.csv"):
        """Initialize analyzer with data file"""
        self.data_file = DATA_DIR / data_file
        self.df = None
        self.fe_result = None
        self.re_result = None
        self.twoway_fe_result = None
        self.hausman_test_result = None
        
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        print("=== 패널데이터 로드 및 전처리 ===")
        
        # 데이터 로드
        df_raw = pd.read_csv(self.data_file)
        print(f"원본 데이터: {df_raw.shape}")
        
        # 전처리: COVID-19 이후 데이터만 사용 (2021년 이후)
        df = df_raw[df_raw['연도'] >= 2021].copy()
        print(f"COVID-19 이후 데이터: {df.shape}")
        
        # 0값을 NaN으로 처리
        df['고용률'] = df['고용률'].replace(0, np.nan)
        df['취업자'] = df['취업자'].replace(0, np.nan)
        df['E9_체류자수'] = df['E9_체류자수'].replace(0, np.nan)
        
        # 결측치 제거
        df = df.dropna(subset=['고용률', 'E9_체류자수']).copy()
        print(f"결측치 제거 후: {df.shape}")
        
        # 연도별 집계 (반기별 -> 연도별)
        df_yearly = df.groupby(['시도', '시군구', '연도']).agg({
            '고용률': 'mean',
            'E9_체류자수': 'sum',
            '제조업_종사자수': 'sum',
            '서비스업_종사자수': 'sum',
            '면적': 'first'
        }).reset_index()
        
        print(f"연도별 집계 후: {df_yearly.shape}")
        
        # 패널 변수 생성
        df_yearly['region_id'] = df_yearly['시도'] + "_" + df_yearly['시군구']
        
        # 새로운 변수 생성
        df_yearly['E9_density'] = df_yearly['E9_체류자수'] / df_yearly['면적']
        df_yearly['manufacturing_concentration'] = df_yearly['제조업_종사자수'] / (
            df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수']
        )
        df_yearly['total_employees'] = df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수']
        df_yearly['employee_density'] = df_yearly['total_employees'] / df_yearly['면적']
        
        # 수도권 더미변수
        df_yearly['capital_area'] = df_yearly['시도'].isin(['서울특별시', '경기도', '인천광역시']).astype(int)
        
        # 로그 변환 (0 문제 해결을 위해 log1p 사용)
        df_yearly['E9_density_log'] = np.log1p(df_yearly['E9_density'])
        df_yearly['employee_density_log'] = np.log1p(df_yearly['employee_density'])
        
        # 결측치 및 무한값 처리
        df_yearly = df_yearly.replace([np.inf, -np.inf], np.nan)
        df_yearly = df_yearly.dropna()
        
        print(f"최종 데이터: {df_yearly.shape}")
        print(f"지역 수: {df_yearly['region_id'].nunique()}")
        print(f"연도 범위: {df_yearly['연도'].min()} - {df_yearly['연도'].max()}")
        
        # 패널 구조 확인
        panel_structure = df_yearly.groupby('region_id').size()
        print(f"지역별 관측치 수 분포:")
        print(panel_structure.value_counts().sort_index())
        
        self.df = df_yearly
        return df_yearly
    
    def prepare_panel_data(self):
        """패널분석을 위한 데이터 준비"""
        print("\n=== 패널분석용 데이터 준비 ===")
        
        # MultiIndex 설정 (패널분석용)
        panel_df = self.df.set_index(['region_id', '연도']).copy()
        
        # 종속변수와 독립변수 정의
        dependent_var = '고용률'
        # Fixed Effect에서는 시간 불변 변수 제외, Random Effect에서는 포함
        fe_independent_vars = [
            'E9_density_log',
            'manufacturing_concentration', 
            'employee_density_log'
        ]
        re_independent_vars = [
            'E9_density_log',
            'manufacturing_concentration', 
            'employee_density_log',
            'capital_area'
        ]
        
        print(f"종속변수: {dependent_var}")
        print(f"FE 독립변수: {fe_independent_vars}")
        print(f"RE 독립변수: {re_independent_vars}")
        
        # 변수 기술통계
        print(f"\n=== 변수 기술통계 ===")
        vars_to_describe = [dependent_var] + re_independent_vars
        print(panel_df[vars_to_describe].describe())
        
        return panel_df, dependent_var, fe_independent_vars, re_independent_vars
    
    def run_fixed_effects_model(self, panel_df, dependent_var, independent_vars):
        """고정효과 모형 (Fixed Effects) 추정"""
        print("\n=== 고정효과 모형 (Fixed Effects) 추정 ===")
        
        # 고정효과 모형 추정
        fe_model = PanelOLS(
            dependent=panel_df[dependent_var],
            exog=panel_df[independent_vars],
            entity_effects=True  # 개체 고정효과
        )
        
        self.fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Fixed Effects Model Results:")
        print(self.fe_result.summary)
        
        return self.fe_result
    
    def run_random_effects_model(self, panel_df, dependent_var, independent_vars):
        """확률효과 모형 (Random Effects) 추정"""
        print("\n=== 확률효과 모형 (Random Effects) 추정 ===")
        
        # 확률효과 모형 추정
        re_model = RandomEffects(
            dependent=panel_df[dependent_var],
            exog=panel_df[independent_vars]
        )
        
        self.re_result = re_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Random Effects Model Results:")
        print(self.re_result.summary)
        
        return self.re_result
    
    def run_twoway_fixed_effects_model(self, panel_df, dependent_var, fe_independent_vars):
        """이원 고정효과 모형 (Two-way Fixed Effects) 추정"""
        print("\n=== 이원 고정효과 모형 (Two-way Fixed Effects) 추정 ===")
        
        # 이원 고정효과 모형 추정 (개체 + 시간 고정효과)
        twoway_fe_model = PanelOLS(
            dependent=panel_df[dependent_var],
            exog=panel_df[fe_independent_vars],
            entity_effects=True,  # 개체 고정효과
            time_effects=True     # 시간 고정효과
        )
        
        self.twoway_fe_result = twoway_fe_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Two-way Fixed Effects Model Results:")
        print(self.twoway_fe_result.summary)
        
        return self.twoway_fe_result
    
    def hausman_test(self):
        """Hausman 검정 (FE vs RE 선택)"""
        print("\n=== Hausman 검정 (Fixed vs Random Effects) ===")
        
        if self.fe_result is None or self.re_result is None:
            print("먼저 FE와 RE 모형을 추정해야 합니다.")
            return None
        
        # 계수 차이 계산
        fe_coef = self.fe_result.params
        re_coef = self.re_result.params
        
        # 분산 차이 계산
        fe_var = self.fe_result.cov
        re_var = self.re_result.cov
        
        # 공통 변수만 사용
        common_vars = fe_coef.index.intersection(re_coef.index)
        
        coef_diff = fe_coef[common_vars] - re_coef[common_vars]
        var_diff = fe_var.loc[common_vars, common_vars] - re_var.loc[common_vars, common_vars]
        
        # Hausman 통계량 계산
        try:
            hausman_stat = coef_diff.T @ np.linalg.inv(var_diff) @ coef_diff
            p_value = 1 - stats.chi2.cdf(hausman_stat, df=len(common_vars))
            
            print(f"Hausman 검정 통계량: {hausman_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"자유도: {len(common_vars)}")
            
            if p_value < 0.05:
                print("결론: Fixed Effects 모형을 선택 (p < 0.05)")
                preferred_model = "Fixed Effects"
            else:
                print("결론: Random Effects 모형을 선택 (p >= 0.05)")
                preferred_model = "Random Effects"
            
            self.hausman_test_result = {
                'statistic': hausman_stat,
                'p_value': p_value,
                'df': len(common_vars),
                'preferred_model': preferred_model
            }
            
        except np.linalg.LinAlgError:
            print("Hausman 검정을 계산할 수 없습니다 (행렬 특이점 문제)")
            self.hausman_test_result = None
        
        return self.hausman_test_result
    
    def compare_models(self):
        """모형 비교"""
        print("\n=== 모형 비교 ===")
        
        models = {
            'Fixed Effects': self.fe_result,
            'Random Effects': self.re_result,
            'Two-way Fixed Effects': self.twoway_fe_result
        }
        
        comparison_df = pd.DataFrame()
        
        for name, result in models.items():
            if result is not None:
                comparison_df[name] = [
                    result.rsquared,
                    result.rsquared_within,
                    result.aic if hasattr(result, 'aic') else np.nan,
                    result.bic if hasattr(result, 'bic') else np.nan,
                    result.nobs
                ]
        
        comparison_df.index = ['R-squared', 'Within R-squared', 'AIC', 'BIC', 'Observations']
        
        print("모형 비교표:")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def create_visualizations(self):
        """시각화 생성"""
        print("\n=== 시각화 생성 ===")
        
        # 1. 계수 비교 그래프
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 고정효과 모형 계수
        if self.fe_result is not None:
            fe_coef = self.fe_result.params
            fe_se = self.fe_result.std_errors
            
            axes[0, 0].barh(range(len(fe_coef)), fe_coef.values)
            axes[0, 0].errorbar(fe_coef.values, range(len(fe_coef)), 
                               xerr=1.96*fe_se.values, fmt='none', color='red')
            axes[0, 0].set_yticks(range(len(fe_coef)))
            axes[0, 0].set_yticklabels(fe_coef.index, fontsize=10)
            axes[0, 0].set_title('Fixed Effects 모형 계수')
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 확률효과 모형 계수
        if self.re_result is not None:
            re_coef = self.re_result.params
            re_se = self.re_result.std_errors
            
            axes[0, 1].barh(range(len(re_coef)), re_coef.values)
            axes[0, 1].errorbar(re_coef.values, range(len(re_coef)), 
                               xerr=1.96*re_se.values, fmt='none', color='red')
            axes[0, 1].set_yticks(range(len(re_coef)))
            axes[0, 1].set_yticklabels(re_coef.index, fontsize=10)
            axes[0, 1].set_title('Random Effects 모형 계수')
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 이원 고정효과 모형 계수
        if self.twoway_fe_result is not None:
            twoway_coef = self.twoway_fe_result.params
            twoway_se = self.twoway_fe_result.std_errors
            
            axes[1, 0].barh(range(len(twoway_coef)), twoway_coef.values)
            axes[1, 0].errorbar(twoway_coef.values, range(len(twoway_coef)), 
                               xerr=1.96*twoway_se.values, fmt='none', color='red')
            axes[1, 0].set_yticks(range(len(twoway_coef)))
            axes[1, 0].set_yticklabels(twoway_coef.index, fontsize=10)
            axes[1, 0].set_title('Two-way Fixed Effects 모형 계수')
            axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # R-squared 비교
        if all(result is not None for result in [self.fe_result, self.re_result, self.twoway_fe_result]):
            r2_data = {
                'Fixed Effects': [self.fe_result.rsquared, self.fe_result.rsquared_within],
                'Random Effects': [self.re_result.rsquared, self.re_result.rsquared_within],
                'Two-way FE': [self.twoway_fe_result.rsquared, self.twoway_fe_result.rsquared_within]
            }
            
            x = np.arange(len(r2_data))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, [r2_data[model][0] for model in r2_data], 
                          width, label='Overall R²')
            axes[1, 1].bar(x + width/2, [r2_data[model][1] for model in r2_data], 
                          width, label='Within R²')
            
            axes[1, 1].set_xlabel('모형')
            axes[1, 1].set_ylabel('R-squared')
            axes[1, 1].set_title('모형별 설명력 비교')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(list(r2_data.keys()))
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'panel_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 잔차 분석
        if self.twoway_fe_result is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 적합값 vs 잔차
            fitted_values = self.twoway_fe_result.fitted_values
            residuals = self.twoway_fe_result.resids
            
            axes[0].scatter(fitted_values, residuals, alpha=0.6, s=30)
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('적합값')
            axes[0].set_ylabel('잔차')
            axes[0].set_title('적합값 vs 잔차')
            axes[0].grid(True, alpha=0.3)
            
            # 잔차 히스토그램
            axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('잔차')
            axes[1].set_ylabel('빈도')
            axes[1].set_title('잔차 분포')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'panel_residual_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self):
        """요약 보고서 생성"""
        print("\n" + "="*80)
        print("패널분석 요약 보고서")
        print("="*80)
        
        print(f"\n데이터 개요:")
        print(f"- 총 관측치 수: {len(self.df):,}개")
        print(f"- 분석 지역 수: {self.df['region_id'].nunique():,}개")
        print(f"- 분석 기간: {self.df['연도'].min()}-{self.df['연도'].max()}")
        
        print(f"\n분석 모형:")
        print("1. Fixed Effects (개체 고정효과)")
        print("2. Random Effects (확률효과)")  
        print("3. Two-way Fixed Effects (개체+시간 고정효과)")
        
        if self.hausman_test_result:
            print(f"\nHausman 검정 결과:")
            print(f"- 검정 통계량: {self.hausman_test_result['statistic']:.4f}")
            print(f"- p-value: {self.hausman_test_result['p_value']:.4f}")
            print(f"- 권장 모형: {self.hausman_test_result['preferred_model']}")
        
        print(f"\n주요 결과 (Two-way Fixed Effects 기준):")
        if self.twoway_fe_result is not None:
            coef = self.twoway_fe_result.params
            se = self.twoway_fe_result.std_errors
            tvalues = self.twoway_fe_result.tstats
            pvalues = self.twoway_fe_result.pvalues
            
            for var in coef.index:
                sig = ""
                if pvalues[var] < 0.01:
                    sig = "***"
                elif pvalues[var] < 0.05:
                    sig = "**"
                elif pvalues[var] < 0.1:
                    sig = "*"
                
                print(f"- {var}: {coef[var]:.4f} ({se[var]:.4f}) {sig}")
            
            print(f"\nModel Fit:")
            print(f"- R-squared: {self.twoway_fe_result.rsquared:.4f}")
            print(f"- Within R-squared: {self.twoway_fe_result.rsquared_within:.4f}")
            print(f"- 관측치 수: {self.twoway_fe_result.nobs}")
        
        print(f"\n해석:")
        print("- 내생성 문제를 통제한 패널분석을 통해 더 신뢰할 수 있는 결과를 도출")
        print("- 개체 및 시간 고정효과를 통해 관측되지 않은 이질성을 통제")
        print("- Two-way Fixed Effects가 가장 엄격한 모형으로 인과관계 추론에 유리")

def main():
    """메인 실행 함수"""
    print("패널분석을 통한 고용률 영향요인 분석")
    print("=" * 80)
    
    try:
        # 분석기 초기화
        analyzer = PanelAnalyzer()
        
        # 1. 데이터 로드 및 전처리
        df_yearly = analyzer.load_and_preprocess_data()
        
        # 2. 패널분석용 데이터 준비
        panel_df, dependent_var, fe_independent_vars, re_independent_vars = analyzer.prepare_panel_data()
        
        # 3. Fixed Effects 모형
        fe_result = analyzer.run_fixed_effects_model(panel_df, dependent_var, fe_independent_vars)
        
        # 4. Random Effects 모형
        re_result = analyzer.run_random_effects_model(panel_df, dependent_var, re_independent_vars)
        
        # 5. Two-way Fixed Effects 모형
        twoway_fe_result = analyzer.run_twoway_fixed_effects_model(panel_df, dependent_var, fe_independent_vars)
        
        # 6. Hausman 검정
        hausman_result = analyzer.hausman_test()
        
        # 7. 모형 비교
        comparison = analyzer.compare_models()
        
        # 8. 시각화
        analyzer.create_visualizations()
        
        # 9. 요약 보고서
        analyzer.generate_summary_report()
        
        print(f"\n패널분석 완료!")
        print("생성된 파일:")
        print("- panel_analysis_results.png")
        print("- panel_residual_analysis.png")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()