#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Panel Data Analysis Module - Following Colab Methodology
Colab 방식을 따른 향상된 패널분석 (상호작용 효과 및 한계효과 분석 포함)
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

class EnhancedPanelAnalyzer:
    """Colab 방식을 따른 향상된 패널분석 클래스"""
    
    def __init__(self, data_file="최종_통합데이터_수정_utf-8.csv"):
        """Initialize analyzer with data file"""
        self.data_file = DATA_DIR / data_file
        self.df = None
        self.panel_df = None
        self.model_result = None
        
    def load_and_preprocess_data(self):
        """Colab 방식을 따른 데이터 로드 및 전처리"""
        print("=== Enhanced 패널데이터 로드 및 전처리 (Colab 방식) ===")
        
        # 데이터 로드
        df1 = pd.read_csv(self.data_file)
        print(f"원본 데이터: {df1.shape}")
        
        # 고용률이 0인 경우 결측치로 치환 (Colab 방식)
        df1['고용률'] = df1['고용률'].replace(0, np.nan)
        df1['취업자'] = df1['취업자'].replace(0, np.nan)
        
        # 고용률이 결측인 행은 모두 제거
        df = df1.dropna(subset=['고용률']).copy()
        print(f"결측치 제거 후: {df1.shape} → {df.shape}")
        
        # COVID-19 이후 데이터만 사용 (안정적 기간)
        df = df[df['연도'] >= 2021].copy()
        print(f"COVID-19 이후 데이터: {df.shape}")
        
        # 연도별 집계 (반기별 → 연도별)
        df_yearly = df.groupby(['시도', '시군구', '연도']).agg({
            '고용률': 'mean',
            'E9_체류자수': 'sum',
            '제조업_종사자수': 'sum',
            '서비스업_종사자수': 'sum',
            '면적': 'first'
        }).reset_index()
        
        print(f"연도별 집계 후: {df_yearly.shape}")
        
        # Colab 방식: 로그변환 for E9
        df_yearly['E9_log'] = np.log1p(df_yearly['E9_체류자수'])
        
        # Colab 방식: 산업 구조 변수 생성
        df_yearly['제조업log'] = np.log1p(df_yearly['제조업_종사자수'])
        df_yearly['서비스업log'] = np.log1p(df_yearly['서비스업_종사자수'])
        df_yearly['산업log비율'] = df_yearly['제조업log'] - df_yearly['서비스업log']  # log((제조+1)/(서비스+1))
        
        # Colab 방식: 상호작용 변수
        df_yearly['interaction'] = df_yearly['E9_log'] * df_yearly['산업log비율']
        
        # 추가 변수들
        df_yearly['E9_density'] = df_yearly['E9_체류자수'] / df_yearly['면적']
        df_yearly['total_employees'] = df_yearly['제조업_종사자수'] + df_yearly['서비스업_종사자수']
        df_yearly['employee_density'] = df_yearly['total_employees'] / df_yearly['면적']
        
        # 수도권 더미변수
        df_yearly['capital_area'] = df_yearly['시도'].isin(['서울특별시', '경기도', '인천광역시']).astype(int)
        
        # 결측치 및 무한값 처리
        df_yearly = df_yearly.replace([np.inf, -np.inf], np.nan)
        df_yearly = df_yearly.dropna()
        
        print(f"최종 데이터: {df_yearly.shape}")
        print(f"지역 수: {df_yearly.groupby(['시도', '시군구']).ngroups}")
        print(f"연도 범위: {df_yearly['연도'].min()} - {df_yearly['연도'].max()}")
        
        self.df = df_yearly
        return df_yearly
    
    def prepare_panel_data(self):
        """Colab 방식을 따른 패널 데이터 준비"""
        print("\n=== 패널데이터 준비 (Colab 방식) ===")
        
        # Colab 방식: 시군구만으로 패널 설정 (시도는 제외)
        # 하지만 동명 시군구를 구분하기 위해 시도+시군구 조합 사용
        self.df['region_id'] = self.df['시도'] + "_" + self.df['시군구']
        
        # MultiIndex 설정 (Colab과 유사하게)
        panel_df = self.df.set_index(['region_id', '연도']).copy()
        
        print("주요 변수 기술통계:")
        key_vars = ['고용률', 'E9_log', '산업log비율', 'interaction']
        print(panel_df[key_vars].describe())
        
        self.panel_df = panel_df
        return panel_df
    
    def run_enhanced_panel_model(self):
        """Colab 방식을 따른 Two-way Fixed Effects 모형 (상호작용 포함)"""
        print("\n=== Enhanced Two-way Fixed Effects 모형 (상호작용 포함) ===")
        
        # Colab 방식: from_formula 사용
        mod = PanelOLS.from_formula(
            '고용률 ~ 1 + E9_log + 산업log비율 + interaction + EntityEffects + TimeEffects',
            data=self.panel_df
        )
        
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        print("Enhanced Panel Model Results:")
        print(res.summary)
        
        self.model_result = res
        return res
    
    def calculate_marginal_effects(self):
        """Colab 방식: 한계효과 계산"""
        print("\n=== 한계효과 계산 (조건부 E9 효과) ===")
        
        if self.model_result is None:
            print("먼저 모형을 추정해야 합니다.")
            return None
        
        # 계수 추출
        b = self.model_result.params
        
        # 산업구조 분위수에서 E9_log의 한계효과 계산
        marginal_effects = {}
        for q in [0.25, 0.50, 0.75]:
            Rq = self.panel_df['산업log비율'].quantile(q)
            me = b['E9_log'] + b['interaction'] * Rq
            marginal_effects[f'{int(q*100)}%'] = {'quantile': Rq, 'effect': me}
            print(f"산업log비율 {int(q*100)}분위({Rq:.3f})에서 E9_log의 한계효과: {me:.4f} p.p. per 1% E9 증가")
        
        return marginal_effects
    
    def create_interaction_visualizations(self):
        """Colab 방식: 상호작용 효과 시각화"""
        print("\n=== 상호작용 효과 시각화 ===")
        
        if self.model_result is None:
            print("먼저 모형을 추정해야 합니다.")
            return
        
        # 계수 및 공분산 행렬 추출
        b = self.model_result.params
        V = self.model_result.cov
        
        # 변수명
        name_E9 = 'E9_log'
        name_R = '산업log비율'
        name_int = 'interaction'
        
        beta_E9 = b[name_E9]
        beta_R = b[name_R]
        beta_int = b[name_int]
        
        var_E9 = V.loc[name_E9, name_E9]
        var_int = V.loc[name_int, name_int]
        cov_E9int = V.loc[name_E9, name_int]
        
        # 그리드 설정
        x = np.linspace(self.panel_df[name_E9].quantile(0.05), 
                       self.panel_df[name_E9].quantile(0.95), 50)
        x0 = self.panel_df[name_E9].median()
        
        # 산업log비율 분위수
        Rq_list = [
            ('25%', self.panel_df[name_R].quantile(0.25)),
            ('50%', self.panel_df[name_R].quantile(0.50)),
            ('75%', self.panel_df[name_R].quantile(0.75))
        ]
        
        # 1) 단순기울기 그래프
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 좌측: 단순기울기
        for lab, Rq in Rq_list:
            slope = beta_E9 + beta_int * Rq
            y = slope * (x - x0)
            axes[0].plot(x, y, label=f'산업log비율 {lab} (R={Rq:.3f})')
        
        axes[0].axhline(0, linestyle='--', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('E9_log')
        axes[0].set_ylabel('Δ고용률 (p.p.)')
        axes[0].set_title('상호작용 단순기울기: 산업log비율 분위수별 E9_log 효과')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2) 한계효과 곡선 (연속 R에서의 ∂고용률/∂E9_log)
        R_grid = np.linspace(self.panel_df[name_R].quantile(0.02),
                            self.panel_df[name_R].quantile(0.98), 200)
        
        M = beta_E9 + beta_int * R_grid
        Var_M = var_E9 + (R_grid**2)*var_int + 2*R_grid*cov_E9int
        SE_M = np.sqrt(np.maximum(Var_M, 0))
        
        crit = 1.96  # 95% CI
        LB = M - crit*SE_M
        UB = M + crit*SE_M
        
        axes[1].plot(R_grid, M, linewidth=2, color='blue', label='한계효과')
        axes[1].fill_between(R_grid, LB, UB, alpha=0.2, color='blue', label='95% 신뢰구간')
        axes[1].axhline(0, linestyle='--', linewidth=1, alpha=0.5, color='red')
        axes[1].set_xlabel('산업log비율')
        axes[1].set_ylabel('E9_log의 한계효과 (p.p. per 1%↑ in E9)')
        axes[1].set_title('E9_log 한계효과 vs. 산업log비율 (95% CI)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'enhanced_interaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3) Johnson-Neyman 구간 분석
        self._johnson_neyman_analysis(R_grid, LB, UB)
    
    def _johnson_neyman_analysis(self, R_grid, LB, UB):
        """Johnson-Neyman 구간 분석"""
        print("\n=== Johnson-Neyman 구간 분석 ===")
        
        sign_pos = (LB > 0)
        sign_neg = (UB < 0)
        
        def contiguous_ranges(mask, grid):
            ranges = []
            on = False
            start = None
            for i, m in enumerate(mask):
                if m and not on:
                    on = True
                    start = grid[i]
                if on and (i == len(mask)-1 or not mask[i+1]):
                    end = grid[i]
                    ranges.append((start, end))
                    on = False
            return ranges
        
        pos_ranges = contiguous_ranges(sign_pos, R_grid)
        neg_ranges = contiguous_ranges(sign_neg, R_grid)
        
        print("JN 양(+) 유의 구간(약 95%):", [(round(a,3), round(b,3)) for a,b in pos_ranges])
        print("JN 음(-) 유의 구간(약 95%):", [(round(a,3), round(b,3)) for a,b in neg_ranges])
        
        return pos_ranges, neg_ranges
    
    def compare_with_simple_model(self):
        """단순 모형과 상호작용 모형 비교"""
        print("\n=== 단순 모형 vs 상호작용 모형 비교 ===")
        
        # 단순 모형 (상호작용 없음)
        simple_mod = PanelOLS.from_formula(
            '고용률 ~ 1 + E9_log + 산업log비율 + EntityEffects + TimeEffects',
            data=self.panel_df
        )
        simple_res = simple_mod.fit(cov_type='clustered', cluster_entity=True)
        
        # 비교표 생성
        comparison = pd.DataFrame({
            'Simple Model': [
                simple_res.rsquared,
                simple_res.rsquared_within,
                simple_res.nobs,
                len(simple_res.params)
            ],
            'Interaction Model': [
                self.model_result.rsquared,
                self.model_result.rsquared_within,
                self.model_result.nobs,
                len(self.model_result.params)
            ]
        }, index=['R-squared', 'Within R-squared', 'Observations', 'Parameters'])
        
        print("모형 비교:")
        print(comparison.round(4))
        
        # F-test for interaction term significance
        print(f"\n상호작용 항 계수: {self.model_result.params['interaction']:.4f}")
        print(f"상호작용 항 p-value: {self.model_result.pvalues['interaction']:.4f}")
        
        return simple_res, comparison
    
    def generate_enhanced_summary_report(self):
        """향상된 요약 보고서 생성"""
        print("\n" + "="*80)
        print("Enhanced 패널분석 요약 보고서 (Colab 방식)")
        print("="*80)
        
        print(f"\n데이터 개요:")
        print(f"- 총 관측치 수: {len(self.panel_df):,}개")
        print(f"- 분석 지역 수: {self.panel_df.index.get_level_values(0).nunique():,}개")
        print(f"- 분석 기간: 2021-2023")
        
        print(f"\n분석 모형:")
        print("Two-way Fixed Effects with Interaction")
        print("고용률 ~ E9_log + 산업log비율 + interaction + EntityEffects + TimeEffects")
        
        if self.model_result is not None:
            print(f"\n주요 결과:")
            params = self.model_result.params
            pvalues = self.model_result.pvalues
            
            for var in ['E9_log', '산업log비율', 'interaction']:
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
            print(f"- R-squared: {self.model_result.rsquared:.4f}")
            print(f"- Within R-squared: {self.model_result.rsquared_within:.4f}")
        
        print(f"\n해석:")
        print("- 상호작용 효과를 포함한 더 정교한 패널분석")
        print("- E9 외국인의 효과가 산업 구조에 따라 어떻게 달라지는지 분석")
        print("- 조건부 한계효과를 통한 정책적 함의 도출")

def main():
    """메인 실행 함수"""
    print("Enhanced 패널분석 (Colab 방식)")
    print("=" * 80)
    
    try:
        # 분석기 초기화
        analyzer = EnhancedPanelAnalyzer()
        
        # 1. 데이터 로드 및 전처리 (Colab 방식)
        df_yearly = analyzer.load_and_preprocess_data()
        
        # 2. 패널분석용 데이터 준비
        panel_df = analyzer.prepare_panel_data()
        
        # 3. Enhanced Two-way Fixed Effects 모형 (상호작용 포함)
        model_result = analyzer.run_enhanced_panel_model()
        
        # 4. 한계효과 계산
        marginal_effects = analyzer.calculate_marginal_effects()
        
        # 5. 상호작용 효과 시각화
        analyzer.create_interaction_visualizations()
        
        # 6. 단순 모형과 비교
        simple_result, comparison = analyzer.compare_with_simple_model()
        
        # 7. 향상된 요약 보고서
        analyzer.generate_enhanced_summary_report()
        
        print(f"\nEnhanced 패널분석 완료!")
        print("생성된 파일:")
        print("- enhanced_interaction_analysis.png")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()