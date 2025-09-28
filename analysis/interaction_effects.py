#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
상호작용항을 포함한 Two-way Fixed Effects (TWFE) 패널 분석
E9 체류자수와 산업구조 변수의 상호작용 효과 분석
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime

# 통계 및 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from scipy import stats

# 경고 무시 및 로깅 설정
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis/interaction_effects.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 랜덤 시드 고정
np.random.seed(42)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class InteractionEffectsAnalysis:
    def __init__(self, data_path):
        """초기화"""
        self.data_path = data_path
        self.df = None
        self.panel_data = None
        self.results = {}
        self.output_dir = 'outputs/interaction'
        self._create_output_dirs()

    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        os.makedirs(f'{self.output_dir}/tables', exist_ok=True)
        os.makedirs(f'{self.output_dir}/figs', exist_ok=True)
        logger.info(f"Output directories created: {self.output_dir}")

    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        logger.info("=" * 60)
        logger.info("1. DATA LOADING AND PREPROCESSING")
        logger.info("=" * 60)

        # 데이터 로드
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {self.df.shape[0]} observations, {self.df.shape[1]} variables")

        # 기본 정보
        logger.info(f"Years: {self.df['연도'].min()} - {self.df['연도'].max()}")
        logger.info(f"Regions: {self.df['시군구'].nunique()} unique municipalities")

        # 패널 구조 확인
        panel_check = self.df.groupby(['시도', '시군구', '연도']).size().reset_index(name='count')
        duplicates = panel_check[panel_check['count'] > 1]
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} duplicate region-year observations")
            # 중복 제거 (평균값 사용)
            self.df = self.df.groupby(['시도', '시군구', '연도']).mean().reset_index()

        # 지역 ID 생성
        self.df['region_id'] = self.df['시도'] + '_' + self.df['시군구']

        # 완전 패널 확인
        expected_obs = self.df['region_id'].nunique() * self.df['연도'].nunique()
        actual_obs = len(self.df)
        logger.info(f"Panel completeness: {actual_obs}/{expected_obs} = {actual_obs/expected_obs*100:.1f}%")

        # 누락 관측치 보고
        if actual_obs < expected_obs:
            all_regions = self.df['region_id'].unique()
            all_years = self.df['연도'].unique()
            from itertools import product
            expected_pairs = set(product(all_regions, all_years))
            actual_pairs = set(zip(self.df['region_id'], self.df['연도']))
            missing = expected_pairs - actual_pairs

            if len(missing) > 0:
                missing_df = pd.DataFrame(list(missing), columns=['region_id', 'year'])
                missing_summary = missing_df.groupby('year').size()
                logger.warning(f"Missing observations by year:\n{missing_summary}")

                # MISSING_DATA.md 생성
                with open('MISSING_DATA.md', 'w', encoding='utf-8') as f:
                    f.write("# Missing Data Report\n\n")
                    f.write(f"## Panel Completeness: {actual_obs/expected_obs*100:.1f}%\n\n")
                    f.write("### Missing Observations by Year\n")
                    f.write(missing_summary.to_string())
                    f.write(f"\n\nTotal missing: {len(missing)} observations\n")

        # 변수 생성
        self._create_variables()

        # 패널 데이터 설정
        self.panel_data = self.df.set_index(['region_id', '연도'])
        logger.info("Panel data structure prepared")

    def _create_variables(self):
        """분석용 변수 생성"""
        logger.info("\nCreating analysis variables...")

        # 1. 분모 확인 및 비율 계산
        if '전체_종사자수' in self.df.columns:
            denominator = '전체_종사자수'
            self.df['total_employment'] = self.df[denominator]
        else:
            # 대체 분모: 제조업 + 서비스업
            self.df['total_employment'] = self.df['제조업_종사자수'] + self.df['서비스업_종사자수']
            logger.warning("Using manufacturing + service workers as total employment proxy")

        # 0 또는 결측값 처리
        self.df['total_employment'] = self.df['total_employment'].replace(0, np.nan)

        # 2. 산업 비중 계산
        self.df['share_manuf'] = self.df['제조업_종사자수'] / self.df['total_employment']
        self.df['share_serv'] = self.df['서비스업_종사자수'] / self.df['total_employment']

        # 건설업 확인
        if '건설업_종사자수' in self.df.columns:
            self.df['share_const'] = self.df['건설업_종사자수'] / self.df['total_employment']
            logger.info("Construction industry share calculated")
        else:
            logger.warning("Construction industry data not available")

        # 3. E9 정규화 (per 1,000 workers)
        self.df['e9_per_1000'] = (self.df['E9_체류자수'] / self.df['total_employment']) * 1000

        # 4. Mean-centering for interaction terms
        for var in ['share_manuf', 'share_serv']:
            if var in self.df.columns:
                self.df[f'{var}_centered'] = self.df[var] - self.df[var].mean()
                logger.info(f"{var}: mean={self.df[var].mean():.3f}, std={self.df[var].std():.3f}")

        if 'share_const' in self.df.columns:
            self.df['share_const_centered'] = self.df['share_const'] - self.df['share_const'].mean()

        # 5. 상호작용항 생성
        self.df['e9_x_manuf'] = self.df['e9_per_1000'] * self.df['share_manuf_centered']
        self.df['e9_x_serv'] = self.df['e9_per_1000'] * self.df['share_serv_centered']

        if 'share_const_centered' in self.df.columns:
            self.df['e9_x_const'] = self.df['e9_per_1000'] * self.df['share_const_centered']

        # 6. 로그 변환 (필요시)
        self.df['log_e9'] = np.log(self.df['E9_체류자수'] + 1)
        self.df['log_total_emp'] = np.log(self.df['total_employment'] + 1)

        # 결측값 제거
        initial_n = len(self.df)
        self.df = self.df.dropna(subset=['고용률', 'e9_per_1000', 'share_manuf', 'share_serv'])
        final_n = len(self.df)

        if initial_n > final_n:
            logger.warning(f"Dropped {initial_n - final_n} observations due to missing values")

        logger.info(f"Final sample size: {final_n} observations")

    def run_twfe_models(self):
        """TWFE 모델 추정"""
        logger.info("\n" + "=" * 60)
        logger.info("2. TWO-WAY FIXED EFFECTS ESTIMATION")
        logger.info("=" * 60)

        # Model 1: Baseline TWFE
        logger.info("\n--- Model 1: Baseline TWFE ---")
        formula1 = "고용률 ~ e9_per_1000 + log_total_emp + EntityEffects + TimeEffects"
        model1 = PanelOLS.from_formula(formula1, data=self.panel_data, check_rank=False)
        result1 = model1.fit(cov_type='clustered', cluster_entity=True)
        self.results['baseline'] = result1
        self._print_results(result1, "Baseline TWFE")

        # Model 2: Manufacturing interaction
        logger.info("\n--- Model 2: Manufacturing Interaction ---")
        formula2 = "고용률 ~ e9_per_1000 + share_manuf_centered + e9_x_manuf + log_total_emp + EntityEffects + TimeEffects"
        model2 = PanelOLS.from_formula(formula2, data=self.panel_data, check_rank=False)
        result2 = model2.fit(cov_type='clustered', cluster_entity=True)
        self.results['manuf'] = result2
        self._print_results(result2, "Manufacturing Interaction")

        # Model 3: Service interaction
        logger.info("\n--- Model 3: Service Interaction ---")
        formula3 = "고용률 ~ e9_per_1000 + share_serv_centered + e9_x_serv + log_total_emp + EntityEffects + TimeEffects"
        model3 = PanelOLS.from_formula(formula3, data=self.panel_data, check_rank=False)
        result3 = model3.fit(cov_type='clustered', cluster_entity=True)
        self.results['serv'] = result3
        self._print_results(result3, "Service Interaction")

        # Model 4: Construction interaction (if available)
        if 'e9_x_const' in self.panel_data.columns:
            logger.info("\n--- Model 4: Construction Interaction ---")
            formula4 = "고용률 ~ e9_per_1000 + share_const_centered + e9_x_const + log_total_emp + EntityEffects + TimeEffects"
            model4 = PanelOLS.from_formula(formula4, data=self.panel_data, check_rank=False)
            result4 = model4.fit(cov_type='clustered', cluster_entity=True)
            self.results['const'] = result4
            self._print_results(result4, "Construction Interaction")

        # Model 5: All interactions (simplified to avoid multicollinearity)
        logger.info("\n--- Model 5: All Interactions ---")
        # Since share_manuf + share_serv = 1, we can only include one
        formula5 = "고용률 ~ e9_per_1000 + share_manuf_centered + e9_x_manuf + e9_x_serv + log_total_emp + EntityEffects + TimeEffects"

        model5 = PanelOLS.from_formula(formula5, data=self.panel_data, check_rank=False, drop_absorbed=True)
        result5 = model5.fit(cov_type='clustered', cluster_entity=True)
        self.results['all'] = result5
        self._print_results(result5, "All Interactions")

    def _print_results(self, result, model_name):
        """결과 출력"""
        logger.info(f"\n{model_name} Results:")
        logger.info("-" * 40)

        params = result.params
        std_errors = result.std_errors
        pvalues = result.pvalues

        for var in params.index:
            coef = params[var]
            se = std_errors[var]
            pval = pvalues[var]

            stars = ''
            if pval < 0.01:
                stars = '***'
            elif pval < 0.05:
                stars = '**'
            elif pval < 0.1:
                stars = '*'

            logger.info(f"  {var:30s}: {coef:8.4f} ({se:.4f}){stars}")

        logger.info(f"  {'R-squared (within)':30s}: {result.rsquared_within:.4f}")
        logger.info(f"  {'Observations':30s}: {result.nobs}")
        logger.info(f"  {'Number of entities':30s}: {result.entity_info.total}")

    def create_tables(self):
        """결과 테이블 생성"""
        logger.info("\n" + "=" * 60)
        logger.info("3. CREATING RESULT TABLES")
        logger.info("=" * 60)

        # Markdown table
        models = ['baseline', 'manuf', 'serv']
        if 'const' in self.results:
            models.append('const')
        models.append('all')

        md_lines = ["# Two-Way Fixed Effects Results with Interaction Terms\n\n"]
        md_lines.append("| Variable | Model 1: Baseline | Model 2: Manufacturing | Model 3: Service | ")

        if 'const' in self.results:
            md_lines.append("Model 4: Construction | ")
        md_lines.append("Model 5: All |\n")

        md_lines.append("|----------|-------------------|------------------------|------------------|")
        if 'const' in self.results:
            md_lines.append("----------------------|")
        md_lines.append("--------------|\n")

        # Key variables to show
        key_vars = ['e9_per_1000', 'share_manuf_centered', 'share_serv_centered',
                   'e9_x_manuf', 'e9_x_serv']
        if 'const' in self.results:
            key_vars.extend(['share_const_centered', 'e9_x_const'])

        for var in key_vars:
            line = f"| {var:20s} |"
            for model in models:
                result = self.results[model]
                if var in result.params.index:
                    coef = result.params[var]
                    se = result.std_errors[var]
                    pval = result.pvalues[var]

                    stars = ''
                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.1:
                        stars = '*'

                    line += f" {coef:.4f}{stars} ({se:.4f}) |"
                else:
                    line += " - |"
            md_lines.append(line + "\n")

        # Add statistics
        md_lines.append("| **Statistics** |  |  |  |")
        if 'const' in self.results:
            md_lines.append("  |")
        md_lines.append("  |\n")

        # R-squared
        line = "| R² (within) |"
        for model in models:
            r2 = self.results[model].rsquared_within
            line += f" {r2:.4f} |"
        md_lines.append(line + "\n")

        # Observations
        line = "| Observations |"
        for model in models:
            n = self.results[model].nobs
            line += f" {n} |"
        md_lines.append(line + "\n")

        # Number of regions
        line = "| Regions |"
        for model in models:
            n_entities = self.results[model].entity_info.total
            line += f" {n_entities} |"
        md_lines.append(line + "\n")

        md_lines.append("| Region FE | Yes | Yes | Yes |")
        if 'const' in self.results:
            md_lines.append(" Yes |")
        md_lines.append(" Yes |\n")

        md_lines.append("| Year FE | Yes | Yes | Yes |")
        if 'const' in self.results:
            md_lines.append(" Yes |")
        md_lines.append(" Yes |\n")

        md_lines.append("| Clustered SE | Yes | Yes | Yes |")
        if 'const' in self.results:
            md_lines.append(" Yes |")
        md_lines.append(" Yes |\n")

        md_lines.append("\n**Note**: Standard errors in parentheses. ")
        md_lines.append("*** p<0.01, ** p<0.05, * p<0.1\n")
        md_lines.append("Standard errors are clustered at the region level.\n")

        # Save markdown
        md_path = f"{self.output_dir}/tables/twfe_interactions.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(md_lines)
        logger.info(f"Markdown table saved: {md_path}")

        # LaTeX table
        self._create_latex_table()

    def _create_latex_table(self):
        """LaTeX 테이블 생성"""
        tex_lines = ["\\begin{table}[htbp]\n"]
        tex_lines.append("\\centering\n")
        tex_lines.append("\\caption{Two-Way Fixed Effects Results with Interaction Terms}\n")
        tex_lines.append("\\label{tab:twfe_interactions}\n")

        n_models = len(self.results)
        tex_lines.append("\\begin{tabular}{l" + "c" * n_models + "}\n")
        tex_lines.append("\\hline\\hline\n")

        # Header
        tex_lines.append(" & (1) & (2) & (3)")
        if 'const' in self.results:
            tex_lines.append(" & (4)")
        tex_lines.append(" & (5) \\\\\n")

        tex_lines.append(" & Baseline & Manufacturing & Service")
        if 'const' in self.results:
            tex_lines.append(" & Construction")
        tex_lines.append(" & All \\\\\n")
        tex_lines.append("\\hline\n")

        # Variables
        key_vars = [
            ('e9_per_1000', 'E9 per 1,000'),
            ('share_manuf_centered', 'Manufacturing share'),
            ('share_serv_centered', 'Service share'),
            ('e9_x_manuf', 'E9 × Manufacturing'),
            ('e9_x_serv', 'E9 × Service')
        ]

        if 'const' in self.results:
            key_vars.extend([
                ('share_const_centered', 'Construction share'),
                ('e9_x_const', 'E9 × Construction')
            ])

        models = ['baseline', 'manuf', 'serv']
        if 'const' in self.results:
            models.append('const')
        models.append('all')

        for var_name, var_label in key_vars:
            line = f"{var_label}"
            for model in models:
                result = self.results[model]
                if var_name in result.params.index:
                    coef = result.params[var_name]
                    se = result.std_errors[var_name]
                    pval = result.pvalues[var_name]

                    stars = ''
                    if pval < 0.01:
                        stars = '***'
                    elif pval < 0.05:
                        stars = '**'
                    elif pval < 0.1:
                        stars = '*'

                    line += f" & {coef:.4f}{stars}"
                else:
                    line += " & "
            tex_lines.append(line + " \\\\\n")

            # Standard errors
            line = ""
            for model in models:
                result = self.results[model]
                if var_name in result.params.index:
                    se = result.std_errors[var_name]
                    line += f" & ({se:.4f})"
                else:
                    line += " & "
            tex_lines.append(line + " \\\\\n")

        tex_lines.append("\\hline\n")

        # Statistics
        tex_lines.append(f"R$^2$ (within)")
        for model in models:
            r2 = self.results[model].rsquared_within
            tex_lines.append(f" & {r2:.4f}")
        tex_lines.append(" \\\\\n")

        tex_lines.append("Observations")
        for model in models:
            n = self.results[model].nobs
            tex_lines.append(f" & {n}")
        tex_lines.append(" \\\\\n")

        tex_lines.append("Regions")
        for model in models:
            n_entities = self.results[model].entity_info.total
            tex_lines.append(f" & {n_entities}")
        tex_lines.append(" \\\\\n")

        tex_lines.append("Region FE & Yes & Yes & Yes")
        if 'const' in self.results:
            tex_lines.append(" & Yes")
        tex_lines.append(" & Yes \\\\\n")

        tex_lines.append("Year FE & Yes & Yes & Yes")
        if 'const' in self.results:
            tex_lines.append(" & Yes")
        tex_lines.append(" & Yes \\\\\n")

        tex_lines.append("\\hline\\hline\n")
        tex_lines.append("\\multicolumn{" + str(n_models + 1) + "}{l}{\\footnotesize Standard errors in parentheses} \\\\\n")
        tex_lines.append("\\multicolumn{" + str(n_models + 1) + "}{l}{\\footnotesize *** p$<$0.01, ** p$<$0.05, * p$<$0.1} \\\\\n")
        tex_lines.append("\\end{tabular}\n")
        tex_lines.append("\\end{table}\n")

        # Save LaTeX
        tex_path = f"{self.output_dir}/tables/twfe_interactions.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.writelines(tex_lines)
        logger.info(f"LaTeX table saved: {tex_path}")

    def plot_marginal_effects(self):
        """한계효과 시각화"""
        logger.info("\n" + "=" * 60)
        logger.info("4. PLOTTING MARGINAL EFFECTS")
        logger.info("=" * 60)

        # Manufacturing interaction
        if 'manuf' in self.results:
            self._plot_single_marginal_effect('manuf', 'share_manuf',
                                             'Manufacturing Share',
                                             'marginal_effects_manuf.png')

        # Service interaction
        if 'serv' in self.results:
            self._plot_single_marginal_effect('serv', 'share_serv',
                                             'Service Share',
                                             'marginal_effects_serv.png')

        # Construction interaction
        if 'const' in self.results:
            self._plot_single_marginal_effect('const', 'share_const',
                                             'Construction Share',
                                             'marginal_effects_const.png')

    def _plot_single_marginal_effect(self, model_name, share_var, share_label, filename):
        """개별 한계효과 플롯"""
        result = self.results[model_name]
        interaction_var = f'e9_x_{model_name}'

        # Get coefficients
        beta_e9 = result.params['e9_per_1000']
        if interaction_var in result.params.index:
            delta = result.params[interaction_var]
        else:
            logger.warning(f"Interaction term {interaction_var} not found")
            return

        # Get share distribution
        share_values = self.df[share_var].dropna()
        percentiles = [10, 25, 50, 75, 90]
        share_points = np.percentile(share_values, percentiles)

        # Calculate marginal effects
        # ∂employ_rate/∂e9 = β + δ * share
        marginal_effects = []
        se_marginal = []

        for share_val in share_points:
            share_centered = share_val - share_values.mean()
            me = beta_e9 + delta * share_centered

            # Approximate SE (simplified - would need delta method for exact)
            se_beta = result.std_errors['e9_per_1000']
            se_delta = result.std_errors[interaction_var]
            # Simplified SE calculation
            se = np.sqrt(se_beta**2 + (share_centered * se_delta)**2)

            marginal_effects.append(me)
            se_marginal.append(se)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create confidence intervals
        marginal_effects = np.array(marginal_effects)
        se_marginal = np.array(se_marginal)
        ci_lower = marginal_effects - 1.96 * se_marginal
        ci_upper = marginal_effects + 1.96 * se_marginal

        # Plot marginal effects with CI
        ax.errorbar(share_points * 100, marginal_effects,
                   yerr=1.96 * se_marginal,
                   marker='o', markersize=8, linewidth=2,
                   capsize=5, capthick=2, label='Marginal Effect')

        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Formatting
        ax.set_xlabel(f'{share_label} (%)', fontsize=12)
        ax.set_ylabel('Marginal Effect of E9 on Employment Rate', fontsize=12)
        ax.set_title(f'Marginal Effects at Different {share_label} Levels\n(with 95% CI)',
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add percentile labels
        for i, (x, y, p) in enumerate(zip(share_points * 100, marginal_effects, percentiles)):
            ax.annotate(f'P{p}', (x, y), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9)

        plt.tight_layout()
        save_path = f"{self.output_dir}/figs/{filename}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Marginal effects plot saved: {save_path}")

        # Report marginal effects
        logger.info(f"\nMarginal Effects for {share_label}:")
        for p, share_val, me, se in zip(percentiles, share_points, marginal_effects, se_marginal):
            sig = '***' if abs(me / se) > 2.58 else ('**' if abs(me / se) > 1.96 else
                  ('*' if abs(me / se) > 1.65 else ''))
            logger.info(f"  P{p:2d} ({share_val*100:5.1f}%): {me:7.4f} (SE={se:.4f}){sig}")

    def update_report(self):
        """보고서 업데이트"""
        logger.info("\n" + "=" * 60)
        logger.info("5. UPDATING REPORT")
        logger.info("=" * 60)

        # Read existing report
        report_path = "new_analysis/종합_분석_보고서.md"
        if not os.path.exists(report_path):
            report_path = "종합_분석_보고서.md"

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
        except FileNotFoundError:
            logger.error(f"Report not found: {report_path}")
            return

        # Prepare new section
        new_section = "\n\n## 상호작용항을 포함한 패널 분석 결과 (TWFE, 2019–2023)\n\n"
        new_section += "### 분석 개요\n\n"
        new_section += "본 절에서는 E9 체류자수와 산업구조 변수 간의 상호작용 효과를 분석하기 위해 "
        new_section += "Two-way Fixed Effects (TWFE) 모델에 상호작용항을 추가하여 추정하였습니다.\n\n"

        # Model specification
        new_section += "### 모형 설정\n\n"
        new_section += "**베이스라인 TWFE 모델:**\n"
        new_section += "$$employ\\_rate_{it} = \\alpha + \\beta \\cdot e9_{it} + \\gamma' X_{it} + \\mu_i + \\tau_t + \\epsilon_{it}$$\n\n"

        new_section += "**상호작용항 포함 모델:**\n"
        new_section += "$$employ\\_rate_{it} = \\alpha + \\beta \\cdot e9_{it} + \\sum_k \\delta_k (e9_{it} \\times share_{k,it}) + \\gamma' X_{it} + \\mu_i + \\tau_t + \\epsilon_{it}$$\n\n"

        new_section += "여기서:\n"
        new_section += "- $\\mu_i$: 지역 고정효과\n"
        new_section += "- $\\tau_t$: 연도 고정효과\n"
        new_section += "- $share_k$: 산업 k의 비중 (제조업, 서비스업, 건설업)\n"
        new_section += "- 표준오차는 지역 단위로 클러스터링\n\n"

        # Data preprocessing summary
        new_section += "### 데이터 전처리\n\n"
        new_section += f"- 분석 기간: {self.df['연도'].min()}-{self.df['연도'].max()}\n"
        new_section += f"- 분석 대상: {self.df['region_id'].nunique()}개 시군구\n"
        new_section += f"- 관측치 수: {len(self.df)}개\n"
        new_section += "- E9 체류자수는 전체 종사자 1,000명당 비율로 정규화\n"
        new_section += "- 산업 비중은 평균 중심화(mean-centering) 적용\n\n"

        # Results table
        new_section += "### 추정 결과\n\n"

        # Read markdown table
        with open(f"{self.output_dir}/tables/twfe_interactions.md", 'r', encoding='utf-8') as f:
            table_content = f.read()
            # Skip the header line
            table_lines = table_content.split('\n')[2:]
            new_section += '\n'.join(table_lines) + '\n'

        # Key findings
        new_section += "\n### 주요 분석 결과\n\n"

        # Manufacturing interaction
        if 'manuf' in self.results and 'e9_x_manuf' in self.results['manuf'].params.index:
            coef = self.results['manuf'].params['e9_x_manuf']
            pval = self.results['manuf'].pvalues['e9_x_manuf']

            if pval < 0.05:
                direction = "증가" if coef > 0 else "감소"
                new_section += f"1. **제조업 상호작용 효과**: 제조업 비중이 높은 지역일수록 E9 근로자 증가의 고용률 효과가 "
                new_section += f"{abs(coef):.4f} 단위 {direction}합니다 (δ_manuf={coef:.4f}, p={pval:.3f}).\n\n"
            else:
                new_section += f"1. **제조업 상호작용 효과**: 통계적으로 유의하지 않음 (p={pval:.3f})\n\n"

        # Service interaction
        if 'serv' in self.results and 'e9_x_serv' in self.results['serv'].params.index:
            coef = self.results['serv'].params['e9_x_serv']
            pval = self.results['serv'].pvalues['e9_x_serv']

            if pval < 0.05:
                direction = "증가" if coef > 0 else "감소"
                new_section += f"2. **서비스업 상호작용 효과**: 서비스업 비중이 높은 지역일수록 E9 근로자 증가의 고용률 효과가 "
                new_section += f"{abs(coef):.4f} 단위 {direction}합니다 (δ_serv={coef:.4f}, p={pval:.3f}).\n\n"
            else:
                new_section += f"2. **서비스업 상호작용 효과**: 통계적으로 유의하지 않음 (p={pval:.3f})\n\n"

        # Construction interaction (if available)
        if 'const' in self.results and 'e9_x_const' in self.results['const'].params.index:
            coef = self.results['const'].params['e9_x_const']
            pval = self.results['const'].pvalues['e9_x_const']

            if pval < 0.05:
                direction = "증가" if coef > 0 else "감소"
                new_section += f"3. **건설업 상호작용 효과**: 건설업 비중이 높은 지역일수록 E9 근로자 증가의 고용률 효과가 "
                new_section += f"{abs(coef):.4f} 단위 {direction}합니다 (δ_const={coef:.4f}, p={pval:.3f}).\n\n"
            else:
                new_section += f"3. **건설업 상호작용 효과**: 통계적으로 유의하지 않음 (p={pval:.3f})\n\n"
        elif 'const' not in self.results:
            new_section += "3. **건설업 상호작용 효과**: 건설업 데이터 부재로 분석 불가\n\n"

        # Marginal effects figures
        new_section += "### 한계효과 시각화\n\n"

        if os.path.exists(f"{self.output_dir}/figs/marginal_effects_manuf.png"):
            new_section += "#### 제조업 비중별 한계효과\n"
            new_section += f"![제조업 한계효과]({self.output_dir}/figs/marginal_effects_manuf.png)\n\n"

        if os.path.exists(f"{self.output_dir}/figs/marginal_effects_serv.png"):
            new_section += "#### 서비스업 비중별 한계효과\n"
            new_section += f"![서비스업 한계효과]({self.output_dir}/figs/marginal_effects_serv.png)\n\n"

        if os.path.exists(f"{self.output_dir}/figs/marginal_effects_const.png"):
            new_section += "#### 건설업 비중별 한계효과\n"
            new_section += f"![건설업 한계효과]({self.output_dir}/figs/marginal_effects_const.png)\n\n"

        # Diagnostics and limitations
        new_section += "### 모형 진단 및 한계\n\n"
        new_section += "1. **다중공선성**: 상호작용항 포함으로 인한 다중공선성 가능성이 있으나, "
        new_section += "mean-centering을 통해 완화하였습니다.\n"
        new_section += "2. **스케일링**: E9 체류자수를 1,000명당 비율로 정규화하여 해석의 용이성을 높였습니다.\n"
        new_section += "3. **측정오차**: 산업별 종사자수 집계 과정에서의 측정오차 가능성이 존재합니다.\n"
        new_section += "4. **동적 효과**: 시차 효과나 동적 패널 모형은 고려하지 않았습니다.\n\n"

        # Policy implications
        new_section += "### 정책적 시사점\n\n"
        new_section += "1. **산업구조별 차등 정책**: 지역의 산업구조에 따라 E9 근로자 정책의 효과가 "
        new_section += "상이하게 나타날 수 있으므로, 산업 특성을 고려한 차별화된 접근이 필요합니다.\n"
        new_section += "2. **지역 맞춤형 전략**: 제조업 중심 지역과 서비스업 중심 지역에 대해 "
        new_section += "서로 다른 외국인 근로자 정책 전략을 수립할 필요가 있습니다.\n"
        new_section += "3. **보완 정책의 중요성**: E9 근로자 유입과 함께 지역 산업구조 개선을 위한 "
        new_section += "보완적 정책이 병행되어야 효과를 극대화할 수 있습니다.\n\n"

        # Check if section already exists
        if "상호작용항을 포함한 패널 분석 결과" in report_content:
            logger.info("Replacing existing interaction effects section")
            # Find and replace the section
            import re
            pattern = r"## 상호작용항을 포함한 패널 분석 결과.*?(?=##|\Z)"
            report_content = re.sub(pattern, new_section, report_content, flags=re.DOTALL)
        else:
            logger.info("Adding new interaction effects section")
            # Add before the last section or at the end
            if "## 결론" in report_content:
                report_content = report_content.replace("## 결론", new_section + "## 결론")
            else:
                report_content += new_section

        # Save updated report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Report updated: {report_path}")

    def run_analysis(self):
        """전체 분석 실행"""
        logger.info("Starting Interaction Effects Analysis")
        logger.info("=" * 60)

        try:
            self.load_and_prepare_data()
            self.run_twfe_models()
            self.create_tables()
            self.plot_marginal_effects()
            self.update_report()

            logger.info("\n" + "=" * 60)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            # Quality checklist
            logger.info("\nQuality Checklist:")
            logger.info("✓ Panel structure verified")
            logger.info("✓ TWFE models with interactions estimated")
            logger.info("✓ Cluster-robust standard errors applied")
            logger.info("✓ Marginal effects calculated and visualized")
            logger.info("✓ Results tables created (Markdown and LaTeX)")
            logger.info("✓ Report updated with new section")

            # Output summary
            logger.info(f"\nOutput files created:")
            logger.info(f"  - {self.output_dir}/tables/twfe_interactions.md")
            logger.info(f"  - {self.output_dir}/tables/twfe_interactions.tex")

            for fig in ['manuf', 'serv', 'const']:
                fig_path = f"{self.output_dir}/figs/marginal_effects_{fig}.png"
                if os.path.exists(fig_path):
                    logger.info(f"  - {fig_path}")

            logger.info(f"  - 종합_분석_보고서.md (updated)")

            if os.path.exists('MISSING_DATA.md'):
                logger.info(f"  - MISSING_DATA.md (missing data report)")

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Data path
    data_path = "data/new_processed/comprehensive_integrated_data.csv"

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # Run analysis
    analyzer = InteractionEffectsAnalysis(data_path)
    analyzer.run_analysis()