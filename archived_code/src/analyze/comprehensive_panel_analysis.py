#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Panel Data Analysis Module - Full Dataset Analysis
전체 데이터(2019-2023)를 활용한 완전한 패널분석 (불균형 패널 포함)

⚠️  ARCHIVED CODE ⚠️
이 파일은 아카이브되었으며 현재 사용되지 않습니다.
메인 분석은 프로젝트 루트의 comprehensive_employment_analysis.py를 사용하세요.

경로 수정이 필요한 경우:
- ROOT_DIR = Path(__file__).parent.parent.parent.parent  (archived_code/src/에서 프로젝트 루트까지)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from linearmodels.panel import PanelOLS, RandomEffects
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

class ComprehensivePanelAnalyzer:
    """전체 데이터를 활용한 포괄적 패널분석 클래스"""
    
    def __init__(self, data_file="최종_통합데이터_수정_utf-8.csv"):
        """Initialize analyzer with data file"""
        self.data_file = DATA_DIR / data_file
        self.df = None
        self.panel_df = None
        self.fe_result = None
        self.re_result = None
        self.twoway_fe_result = None
        self.enhanced_result = None
        
    def load_and_preprocess_full_data(self):
        """전체 데이터 로드 및 전처리 (모든 관측치 활용)"""
        print("=== 전체 데이터 포괄적 패널분석 (2019-2023) ===")
        
        # 데이터 로드
        df_raw = pd.read_csv(self.data_file)
        print(f"원본 데이터: {df_raw.shape}")
        
        # 기본 정보 출력
        print(f"\n=== 전체 데이터 개요 ===")
        print(f"총 관측치: {len(df_raw):,}개")
        print(f"연도 범위: {df_raw['연도'].min()} - {df_raw['연도'].max()}")
        print(f"지역 수: {df_raw.groupby(['시도', '시군구']).ngroups:,}개")
        print(f"시도 수: {df_raw['시도'].nunique()}개")
        
        print(f"\n연도별 분포:")
        print(df_raw['연도'].value_counts().sort_index())
        
        # 고용률 0값 처리 전략 수정: 0을 실제 값으로 유지
        df = df_raw.copy()
        
        # 단, 명백한 결측치는 제거 (NaN 값들)
        initial_rows = len(df)
        df = df.dropna(subset=['고용률', 'E9_체류자수']).copy()
        print(f"명시적 결측치 제거: {initial_rows:,} → {len(df):,} 행")
        
        # 연도별 집계 (반기별 → 연도별) - 모든 데이터 유지
        print(f"\n=== 연도별 집계 ===")
        df_yearly = df.groupby(['시도', '시군구', '연도']).agg({
            '고용률': 'mean',  # 0값도 포함하여 평균
            'E9_체류자수': 'sum',
            '제조업_종사자수': 'sum', 
            '서비스업_종사자수': 'sum',
            '면적': 'first'
        }).reset_index()
        
        print(f"연도별 집계 후: {len(df_yearly):,}행")
        
        # 지역 ID 생성
        df_yearly['region_id'] = df_yearly['시도'] + "_" + df_yearly['시군구']
        
        # 새로운 변수 생성
        print(f"\n=== 변수 생성 ===")
        
        # COVID-19 더미변수 (구조적 변화 통제)
        df_yearly['covid_period'] = (df_yearly['연도'] <= 2020).astype(int)
        df_yearly['post_covid'] = (df_yearly['연도'] >= 2021).astype(int)
        
        # E9 관련 변수
        df_yearly['E9_log'] = np.log1p(df_yearly['E9_체류자수'])
        df_yearly['E9_density'] = df_yearly['E9_체류자수'] / df_yearly['면적']
        df_yearly['E9_density_log'] = np.log1p(df_yearly['E9_density'])
        
        # 산업 구조 변수
        df_yearly['제조업log'] = np.log1p(df_yearly['제조업_종사자수'])
        df_yearly['서비스업log'] = np.log1p(df_yearly['서비스업_종사자수'])
        df_yearly['산업log비율'] = df_yearly['제조업log'] - df_yearly['서비스업log']
        
        # 제조업 집중도 (0으로 나누기 방지)
        total_employees = df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수']
        df_yearly['manufacturing_concentration'] = np.where(
            total_employees > 0,
            df_yearly['제조업_종사자수'] / total_employees,
            0.5  # 기본값: 균등 분배
        )
        
        # 종사자 밀도
        df_yearly['total_employees'] = total_employees
        df_yearly['employee_density'] = df_yearly['total_employees'] / df_yearly['면적']
        df_yearly['employee_density_log'] = np.log1p(df_yearly['employee_density'])
        
        # 상호작용 변수
        df_yearly['interaction'] = df_yearly['E9_log'] * df_yearly['산업log비율']
        df_yearly['covid_E9_interaction'] = df_yearly['covid_period'] * df_yearly['E9_log']
        
        # 수도권 더미변수
        df_yearly['capital_area'] = df_yearly['시도'].isin(['서울특별시', '경기도', '인천광역시']).astype(int)
        
        # 무한값 처리
        df_yearly = df_yearly.replace([np.inf, -np.inf], np.nan)
        df_yearly = df_yearly.dropna()
        
        print(f"최종 데이터: {len(df_yearly):,}행, {df_yearly.shape[1]}개 변수")
        print(f"지역 수: {df_yearly['region_id'].nunique():,}개")
        print(f"평균 관측 시점: {len(df_yearly) / df_yearly['region_id'].nunique():.1f}년")
        
        # 패널 구조 분석
        panel_structure = df_yearly.groupby('region_id').size()
        print(f"\n=== 패널 구조 ===")
        print("지역별 관측 시점 분포:")
        print(panel_structure.value_counts().sort_index())
        
        # 균형 vs 불균형 패널
        balanced_regions = (panel_structure == 5).sum()
        unbalanced_regions = (panel_structure < 5).sum()
        print(f"균형패널 지역: {balanced_regions}개")
        print(f"불균형패널 지역: {unbalanced_regions}개")
        
        self.df = df_yearly
        return df_yearly
    
    def prepare_comprehensive_panel_data(self):
        """포괄적 패널 데이터 준비"""
        print("\n=== 포괄적 패널데이터 준비 ===")
        
        # MultiIndex 설정 (불균형 패널 허용)
        panel_df = self.df.set_index(['region_id', '연도']).copy()
        
        print("주요 변수 기술통계 (전체 기간):")
        key_vars = ['고용률', 'E9_log', '산업log비율', 'interaction', 'covid_period', 'post_covid']
        print(panel_df[key_vars].describe())
        
        # 연도별 통계
        print(f"\n연도별 주요 변수 평균:")
        yearly_stats = self.df.groupby('연도')[['고용률', 'E9_체류자수', '제조업_종사자수', '서비스업_종사자수']].mean()
        print(yearly_stats.round(2))
        
        self.panel_df = panel_df
        return panel_df
    
    def run_comprehensive_fixed_effects(self):
        """포괄적 고정효과 모형 (전체 기간)"""
        print("\n=== 포괄적 Fixed Effects 모형 (2019-2023) ===")
        
        # 기본 FE 모형 (COVID 더미 포함)
        fe_vars = ['E9_density_log', 'manufacturing_concentration', 'employee_density_log', 'covid_period']
        
        fe_model = PanelOLS(
            dependent=self.panel_df['고용률'],
            exog=self.panel_df[fe_vars],
            entity_effects=True
        )
        
        self.fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Comprehensive Fixed Effects Results:")
        print(self.fe_result.summary)
        
        return self.fe_result
    
    def run_comprehensive_random_effects(self):
        """포괄적 확률효과 모형"""
        print("\n=== 포괄적 Random Effects 모형 ===")
        
        # RE 모형 (시간 불변 변수 포함 가능)
        re_vars = ['E9_density_log', 'manufacturing_concentration', 'employee_density_log', 'covid_period', 'capital_area']
        
        re_model = RandomEffects(
            dependent=self.panel_df['고용률'],
            exog=self.panel_df[re_vars]
        )
        
        self.re_result = re_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Comprehensive Random Effects Results:")
        print(self.re_result.summary)
        
        return self.re_result
    
    def run_comprehensive_twoway_fe(self):
        """포괄적 이원 고정효과 모형"""
        print("\n=== 포괄적 Two-way Fixed Effects 모형 ===")
        
        # Two-way FE 모형 (개체 + 시간 고정효과)
        twoway_vars = ['E9_density_log', 'manufacturing_concentration', 'employee_density_log']
        
        twoway_fe_model = PanelOLS(
            dependent=self.panel_df['고용률'],
            exog=self.panel_df[twoway_vars],
            entity_effects=True,
            time_effects=True
        )
        
        self.twoway_fe_result = twoway_fe_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Comprehensive Two-way Fixed Effects Results:")
        print(self.twoway_fe_result.summary)
        
        return self.twoway_fe_result
    
    def run_enhanced_comprehensive_model(self):
        """향상된 포괄적 모형 (상호작용 효과)"""
        print("\n=== 향상된 포괄적 모형 (상호작용 효과) ===")
        
        # 향상된 모형: 상호작용 효과 (COVID는 시간고정효과로 통제됨)
        enhanced_model = PanelOLS.from_formula(
            '고용률 ~ 1 + E9_log + 산업log비율 + interaction + EntityEffects + TimeEffects',
            data=self.panel_df
        )
        
        self.enhanced_result = enhanced_model.fit(cov_type='clustered', cluster_entity=True)
        
        print("Enhanced Comprehensive Model Results:")
        print(self.enhanced_result.summary)
        
        return self.enhanced_result
    
    def calculate_comprehensive_marginal_effects(self):
        """포괄적 한계효과 계산"""
        print("\n=== 포괄적 한계효과 분석 ===")
        
        if self.enhanced_result is None:
            print("먼저 향상된 모형을 추정해야 합니다.")
            return None
        
        b = self.enhanced_result.params
        
        # COVID 전후 기본 통계 (시간고정효과로 통제됨)
        print("=== COVID 전후 기본 통계 ===")
        covid_before_data = self.df[self.df['연도'] <= 2020]
        covid_after_data = self.df[self.df['연도'] >= 2021]
        
        print(f"COVID 이전 평균 고용률: {covid_before_data['고용률'].mean():.2f}%")
        print(f"COVID 이후 평균 고용률: {covid_after_data['고용률'].mean():.2f}%")
        print(f"COVID 이전 평균 E9: {covid_before_data['E9_체류자수'].mean():.0f}명")
        print(f"COVID 이후 평균 E9: {covid_after_data['E9_체류자수'].mean():.0f}명")
        
        # 산업구조별 효과 (전체 기간)
        print(f"\n=== 산업구조별 E9 효과 (전체 기간) ===")
        for q in [0.25, 0.50, 0.75]:
            Rq = self.panel_df['산업log비율'].quantile(q)
            me = b['E9_log'] + b['interaction'] * Rq
            print(f"산업log비율 {int(q*100)}분위({Rq:.3f})에서 E9_log 효과: {me:.4f} p.p.")
        
        return {
            'covid_before_rate': covid_before_data['고용률'].mean(),
            'covid_after_rate': covid_after_data['고용률'].mean(),
            'covid_before_e9': covid_before_data['E9_체류자수'].mean(),
            'covid_after_e9': covid_after_data['E9_체류자수'].mean()
        }
    
    def create_comprehensive_visualizations(self):
        """포괄적 시각화 (전체 기간 + COVID 효과)"""
        print("\n=== 포괄적 시각화 생성 ===")
        
        # 1. 모형 비교 + COVID 효과
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 기본 FE 모형 계수
        if self.fe_result is not None:
            fe_coef = self.fe_result.params
            fe_se = self.fe_result.std_errors
            
            axes[0, 0].barh(range(len(fe_coef)), fe_coef.values)
            axes[0, 0].errorbar(fe_coef.values, range(len(fe_coef)), 
                               xerr=1.96*fe_se.values, fmt='none', color='red')
            axes[0, 0].set_yticks(range(len(fe_coef)))
            axes[0, 0].set_yticklabels(fe_coef.index, fontsize=9)
            axes[0, 0].set_title('Fixed Effects 모형 (COVID 통제)')
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].grid(True, alpha=0.3)
        
        # RE 모형 계수
        if self.re_result is not None:
            re_coef = self.re_result.params
            re_se = self.re_result.std_errors
            
            axes[0, 1].barh(range(len(re_coef)), re_coef.values)
            axes[0, 1].errorbar(re_coef.values, range(len(re_coef)), 
                               xerr=1.96*re_se.values, fmt='none', color='red')
            axes[0, 1].set_yticks(range(len(re_coef)))
            axes[0, 1].set_yticklabels(re_coef.index, fontsize=9)
            axes[0, 1].set_title('Random Effects 모형')
            axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Two-way FE 모형 계수
        if self.twoway_fe_result is not None:
            twoway_coef = self.twoway_fe_result.params
            twoway_se = self.twoway_fe_result.std_errors
            
            axes[0, 2].barh(range(len(twoway_coef)), twoway_coef.values)
            axes[0, 2].errorbar(twoway_coef.values, range(len(twoway_coef)), 
                               xerr=1.96*twoway_se.values, fmt='none', color='red')
            axes[0, 2].set_yticks(range(len(twoway_coef)))
            axes[0, 2].set_yticklabels(twoway_coef.index, fontsize=9)
            axes[0, 2].set_title('Two-way Fixed Effects 모형')
            axes[0, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 2].grid(True, alpha=0.3)
        
        # 연도별 고용률 추이
        yearly_employment = self.df.groupby('연도')['고용률'].mean()
        axes[1, 0].plot(yearly_employment.index, yearly_employment.values, 
                       marker='o', linewidth=3, markersize=8, color='blue')
        axes[1, 0].axvline(x=2020, color='red', linestyle='--', alpha=0.7, label='COVID-19')
        axes[1, 0].set_xlabel('연도')
        axes[1, 0].set_ylabel('평균 고용률 (%)')
        axes[1, 0].set_title('연도별 고용률 추이 (전체 기간)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # COVID 전후 비교
        covid_before = self.df[self.df['연도'] <= 2020]['고용률'].mean()
        covid_after = self.df[self.df['연도'] >= 2021]['고용률'].mean()
        
        axes[1, 1].bar(['COVID 이전\n(2019-2020)', 'COVID 이후\n(2021-2023)'], 
                      [covid_before, covid_after], 
                      color=['lightcoral', 'lightblue'], alpha=0.7)
        axes[1, 1].set_ylabel('평균 고용률 (%)')
        axes[1, 1].set_title('COVID 전후 고용률 비교')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 연도별 E9 체류자수 추이
        yearly_e9 = self.df.groupby('연도')['E9_체류자수'].sum()
        axes[1, 2].plot(yearly_e9.index, yearly_e9.values, 
                       marker='s', linewidth=3, markersize=8, color='green')
        axes[1, 2].axvline(x=2020, color='red', linestyle='--', alpha=0.7, label='COVID-19')
        axes[1, 2].set_xlabel('연도')
        axes[1, 2].set_ylabel('총 E9 체류자수')
        axes[1, 2].set_title('연도별 E9 체류자수 추이')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'comprehensive_panel_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_all_models(self):
        """모든 모형 비교"""
        print("\n=== 전체 모형 비교 ===")
        
        models = {
            'Fixed Effects': self.fe_result,
            'Random Effects': self.re_result, 
            'Two-way FE': self.twoway_fe_result,
            'Enhanced Model': self.enhanced_result
        }
        
        comparison_df = pd.DataFrame()
        
        for name, result in models.items():
            if result is not None:
                comparison_df[name] = [
                    result.rsquared if hasattr(result, 'rsquared') else np.nan,
                    result.rsquared_within if hasattr(result, 'rsquared_within') else np.nan,
                    result.nobs if hasattr(result, 'nobs') else np.nan,
                    len(result.params) if hasattr(result, 'params') else np.nan
                ]
        
        comparison_df.index = ['R-squared', 'Within R-squared', 'Observations', 'Parameters']
        
        print("전체 모형 비교표:")
        print(comparison_df.round(4))
        
        return comparison_df
    
    def generate_comprehensive_summary(self):
        """포괄적 요약 보고서"""
        print("\n" + "="*80)
        print("포괄적 패널분석 요약 보고서 (전체 데이터 2019-2023)")
        print("="*80)
        
        print(f"\n데이터 개요:")
        print(f"- 총 관측치: {len(self.df):,}개 (전체 데이터)")
        print(f"- 분석 지역: {self.df['region_id'].nunique():,}개")
        print(f"- 분석 기간: {self.df['연도'].min()}-{self.df['연도'].max()} (5년)")
        print(f"- 패널 형태: 불균형 패널 (모든 관측치 활용)")
        
        # COVID 전후 기초 통계
        covid_before_data = self.df[self.df['연도'] <= 2020]
        covid_after_data = self.df[self.df['연도'] >= 2021]
        
        print(f"\nCOVID 전후 비교:")
        print(f"- COVID 이전 관측치: {len(covid_before_data):,}개")
        print(f"- COVID 이후 관측치: {len(covid_after_data):,}개")
        print(f"- COVID 이전 평균 고용률: {covid_before_data['고용률'].mean():.2f}%")
        print(f"- COVID 이후 평균 고용률: {covid_after_data['고용률'].mean():.2f}%")
        
        print(f"\n분석 모형:")
        print("1. Fixed Effects (COVID 더미 포함)")
        print("2. Random Effects (수도권 더미 포함)")
        print("3. Two-way Fixed Effects (개체+시간 고정효과)")
        print("4. Enhanced Model (상호작용 + COVID 효과)")
        
        if self.enhanced_result is not None:
            print(f"\n주요 결과 (Enhanced Model):")
            params = self.enhanced_result.params
            pvalues = self.enhanced_result.pvalues
            
            key_vars = ['E9_log', '산업log비율', 'interaction']
            for var in key_vars:
                if var in params:
                    sig = ""
                    if pvalues[var] < 0.01:
                        sig = "***"
                    elif pvalues[var] < 0.05:
                        sig = "**"
                    elif pvalues[var] < 0.1:
                        sig = "*"
                    
                    print(f"- {var}: {params[var]:.4f} (p={pvalues[var]:.4f}) {sig}")
            
            print(f"\nModel Fit:")
            print(f"- R-squared: {self.enhanced_result.rsquared:.4f}")
            print(f"- Within R-squared: {self.enhanced_result.rsquared_within:.4f}")
            print(f"- 관측치 수: {self.enhanced_result.nobs}")
        
        print(f"\n정책적 함의:")
        print("- 전체 기간 분석을 통한 더 강건한 결과 도출")
        print("- COVID-19 구조적 변화를 명시적으로 통제")
        print("- 불균형 패널 활용으로 모든 관측치의 정보 활용")
        print("- 지역별, 시기별 차별화된 정책 설계 근거 제공")

def main():
    """메인 실행 함수"""
    print("포괄적 패널분석 (전체 데이터 2019-2023)")
    print("=" * 80)
    
    try:
        # 분석기 초기화
        analyzer = ComprehensivePanelAnalyzer()
        
        # 1. 전체 데이터 로드 및 전처리
        df_full = analyzer.load_and_preprocess_full_data()
        
        # 2. 포괄적 패널 데이터 준비
        panel_df = analyzer.prepare_comprehensive_panel_data()
        
        # 3. Fixed Effects 모형
        fe_result = analyzer.run_comprehensive_fixed_effects()
        
        # 4. Random Effects 모형
        re_result = analyzer.run_comprehensive_random_effects()
        
        # 5. Two-way Fixed Effects 모형
        twoway_result = analyzer.run_comprehensive_twoway_fe()
        
        # 6. 향상된 포괄적 모형
        enhanced_result = analyzer.run_enhanced_comprehensive_model()
        
        # 7. 포괄적 한계효과 분석
        marginal_effects = analyzer.calculate_comprehensive_marginal_effects()
        
        # 8. 포괄적 시각화
        analyzer.create_comprehensive_visualizations()
        
        # 9. 모형 비교
        comparison = analyzer.compare_all_models()
        
        # 10. 포괄적 요약 보고서
        analyzer.generate_comprehensive_summary()
        
        print(f"\n포괄적 패널분석 완료!")
        print("생성된 파일:")
        print("- comprehensive_panel_analysis.png")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()